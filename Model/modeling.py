import pandas as pd
import torch
import random
import numpy as np
import os
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

def load_data(data_path, predict_path):
    '''
    載入前處理好的資料及要預測的帳戶檔案。

    Args:
        data_path (dict): 前處理包裝好的資料。
        predict_path (pd.DataFrame): 待預測帳戶清單。

    Returns:
        data: 包含所有節點特徵、邊索引的 Data 物件。
        accounts: 所有帳戶。
        y: 警示帳戶標籤。
        predict: 待預測帳戶檔案。
    '''
    loaded_dict = torch.load(data_path, weights_only = False)
    data = loaded_dict['data']
    accounts = loaded_dict['accounts']
    y = loaded_dict['y']
    predict = pd.read_csv(predict_path)

    return data, accounts, y, predict

def set_seed(seed):
    '''設定所有必要的隨機種子，降低隨機性。'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class PULoss(torch.nn.Module):
    '''
    基於 cross entropy loss 修正的 Positive-Unlabeled (PU) 概念損失函式。
    在計算未標記資料 loss 部分時，考慮有部分比例的正樣本損失，及去掉這部分比例的負樣本損失。
    
    Args:
        alpha (float): 先驗機率的估計值，即未標記樣本中正類的比例。
    '''
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        '''
        計算 PU 損失。
        
        Args:
            pred (torch.Tensor): 模型的 logit 輸出。
            target (torch.Tensor): 標籤。
            
        Returns:
            loss: 計算出的 PU 損失值。
        '''
        # 使用 softmax 取得類別 1 的機率
        prob_pos = F.softmax(pred, dim = 1)[:, 1]

        # 損失項
        positive_loss = -torch.log(prob_pos + 1e-6)
        negative_loss = -torch.log(1 - prob_pos + 1e-6)

        is_positive = (target == 1)
        is_unlabeled = (target == 0)

        loss = torch.tensor(0.0, device = pred.device)

        # 正類損失
        if is_positive.sum() > 0:
            pos_loss = positive_loss[is_positive].mean()
            loss += pos_loss

        # 未標記損失
        if is_unlabeled.sum() > 0:
            unlabeled_loss = (self.alpha * positive_loss[is_unlabeled] + 
                              (1 - self.alpha) * negative_loss[is_unlabeled]).mean()
            loss += unlabeled_loss

        return loss

class GraphSAGE(torch.nn.Module):
    '''
    多層的 GraphSAGE 模型。
    引入可學習的權重參數，把每一層更新自身節點表示的結果，做加權的特徵融合，再進行分類。

    Args:
        in_channels (int): 輸入特徵維度。
        hidden_channels (int): 隱藏層特徵維度。
        out_channels (int): 輸出類別數。
        num_layers (int): 層數。
        aggr (str): 鄰居聚合方式。
        dropout (float): Dropout 比例。
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 4, aggr = 'mean', dropout = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggr = aggr
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                channels = in_channels
            else:
                channels = hidden_channels
            
            self.convs.append(SAGEConv(channels, hidden_channels, normalize = True, project = True, aggr = self.aggr))
            self.norms.append(torch.nn.LayerNorm(hidden_channels))

        # 可學習權重參數
        self.layer_weights = torch.nn.Parameter(torch.ones(num_layers))
        # 線性分類層
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        '''
        前向傳播。
        
        Args:
            x (torch.Tensor): 節點特徵。
            edge_index (torch.Tensor): 邊索引。
            
        Returns:
            x: 模型的 logit 輸出。
        '''
        hiddens = []

        # 卷積層迭代
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p = self.dropout, training = self.training)
            hiddens.append(x)

        # 根據可學習權重進行加權特徵融合
        weights = F.softmax(self.layer_weights, dim=0)
        x = sum(w * h for w, h in zip(weights, hiddens))
        
        # 分類器輸出
        x = self.classifier(x)
        return x

def training_setups(model, data, device):
    '''
    設定損失函數、優化器等。

    Args:
        model (torch.nn.Module): 待訓練的模型。
        data (Data): 包含所有節點特徵和邊索引的 PyG Data 物件。
        device (torch.device): 執行訓練的運算裝置。
            
    Returns:
        criterion (PULoss): Positive-Unlabeled 損失函式。
        optimizer (optim.Optimizer): 用於更新模型權重的優化器。
        compiled_model (torch.nn.Module): 使用 torch.compile 加速待訓練的模型。
        scaler (GradScaler): 用於混合精度訓練。
    '''
    # 計算 PU 損失函式的先驗參數 alpha
    alert_count = (data.y[data.train_mask] == 1).sum().item()
    unlabel_count = (data.y[data.train_mask] == 0).sum().item()
    prior_alpha = alert_count / unlabel_count
    criterion = PULoss(alpha = prior_alpha)

    optimizer = optim.Adam(model.parameters(), lr = 0.005, weight_decay = 1e-5)

    # 使用 torch.compile 加速訓練
    compiled_model = torch.compile(model, mode = 'reduce-overhead')
    # 初始化混和精度
    scaler = GradScaler()

    return criterion, optimizer, compiled_model, scaler

def train_model(compiled_model, data, criterion, optimizer, scaler, device):
    '''
    執行訓練。

    Args:
        compiled_model (torch.nn.Module): 使用 torch.compile 加速待訓練的模型。
        data (Data): 包含所有節點特徵和邊索引的 PyG Data 物件。
        criterion (PULoss): Positive-Unlabeled 損失函式。
        optimizer (optim.Optimizer): 用於更新模型權重的優化器。
        scaler (GradScaler): 用於混合精度訓練。
        device (torch.device): 執行訓練的運算裝置。
            
    Returns:
        loss.item(): 訓練損失。
    '''
    compiled_model.train()
    optimizer.zero_grad()
    
    with torch.autocast(device_type = device.type, dtype = torch.float16):
        out = compiled_model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
        
    return loss.item()

@torch.no_grad()
def validate_model(model, data, criterion, device):
    '''
    執行驗證。

    Args:
        model (torch.nn.Module): 待驗證的模型。
        data (Data): 包含所有節點特徵和邊索引的 PyG Data 物件。
        criterion (PULoss): Positive-Unlabeled 損失函式。
        device (torch.device): 執行訓練的運算裝置。
    
    Returns:
        val_loss.item(): 驗證損失。
        ap: 驗證 AUPRC。
    '''
    model.eval()
    
    with torch.autocast(device_type = device.type, dtype = torch.float16):
        out = model(data.x, data.edge_index)
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    
    pred_proba = F.softmax(out[data.val_mask].float(), dim = 1)[:, 1] 
    y_true = data.y[data.val_mask].cpu().numpy()
    ap = average_precision_score(y_true, pred_proba.cpu().numpy())

    return val_loss.item(), ap

def training_loop(compiled_model, model, data, criterion, optimizer, scaler, device):
    '''
    執行完整訓練循環。
    
    Args:
        compiled_model (torch.nn.Module): 使用 torch.compile 加速待訓練的模型。
        model (torch.nn.Module): 待驗證的模型。
        data (Data): 包含所有節點特徵和邊索引的 PyG Data 物件。
        criterion (PULoss): Positive-Unlabeled 損失函式。
        optimizer (optim.Optimizer): 用於更新模型權重的優化器。
        scaler (GradScaler): 用於混合精度訓練。
        device (torch.device): 執行訓練的運算裝置。
    '''
    best_auprc = 0

    for epoch in range(1000):
        train_loss = train_model(compiled_model, data, criterion, optimizer, scaler, device)
        val_loss, val_auprc = validate_model(model, data, criterion, device)
        
        print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUPRC: {val_auprc:.4f}')

        # 儲存最佳模型 (基於 Val AUPRC)
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'********SAVE EPOCH:{epoch}, New Best AUPRC: {best_auprc:.4f}')

        if epoch % 100 == 0:
            torch.cuda.empty_cache()

def find_best_threshold(val_proba, val_true):
    '''
    在驗證集上搜尋最佳的 F1 閾值。

    Args:
        val_proba (float): 預測驗證集的機率值。
        val_true (int): 驗證集真實標籤。

    Returns:
        best_threshold: 最佳閾值。
        best_f1: 最佳閾值下的最佳 F1 分數。
    '''
    thresholds = np.arange(0.1, 1, 0.05)
    best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        pred_label = (val_proba >= threshold).astype(int)

        precision = precision_score(val_true, pred_label, zero_division = 0)
        recall = recall_score(val_true, pred_label, zero_division = 0)
        f1 = f1_score(val_true, pred_label, zero_division = 0)
        
        print(f'Threshold: {threshold:.2f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    print(f'最佳閾值: {best_threshold:.2f}')
    print(f'最佳 F1 分數: {best_f1:.4f}')

    return best_threshold, best_f1
    
def evaluate_model(model, data, accounts, acct_test):
    '''
    最佳模型評估及預測。

    Args:
        model (torch.nn.Module): 訓練好待驗證的模型。
        data (Data): 包含所有節點特徵和邊索引的 PyG Data 物件。
        accounts (list): 所有帳戶。
        acct_test (pd.DataFrame): 待預測帳戶清單。

    Returns:
        predict: 測試集預測結果
    '''
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)

    # 取得驗證集機率和標籤
    val_proba = F.softmax(out[data.val_mask], dim = 1)[:, 1].detach().cpu().numpy()
    val_true = data.y[data.val_mask].cpu().numpy()

    # 搜尋最佳 F1 閾值
    best_threshold, best_f1 = find_best_threshold(val_proba, val_true)
    
    # 最終預測與整體 F1 分數報告
    final_pred_labels = (F.softmax(out, dim=1)[:, 1].cpu().numpy() >= best_threshold).astype(int)

    train_mask_np = data.train_mask.cpu().numpy()
    train_true = data.y[data.train_mask].cpu().numpy()
    val_mask_np = data.val_mask.cpu().numpy()
    val_true = data.y[data.val_mask].cpu().numpy()

    train_f1 = f1_score(train_true, final_pred_labels[train_mask_np], zero_division = 0)
    val_f1 = f1_score(val_true, final_pred_labels[val_mask_np], zero_division = 0)
    print(f'Train F1 Score: {train_f1:.4f}')
    print(f'Val F1 Score: {val_f1:.4f}')

    # 產生測試集預測結果
    test_mask_np = data.test_mask.cpu().numpy()
    test_pred = final_pred_labels[test_mask_np]
    accounts_np = np.array(accounts)
    test_accounts_list = accounts_np[test_mask_np].tolist()
    
    predictions = pd.DataFrame({
        'acct': test_accounts_list,
        'prediction': test_pred})
    
    print(f"測試集預測的正類數量: {predictions['prediction'].sum()}")

    # 合併到原始 predict 檔案並儲存
    predict = acct_test.merge(
        predictions,
        on = 'acct',
        how = 'left')
    
    predict.drop('label', axis = 1, inplace = True)
    predict.rename(columns = {'prediction': 'label'}, inplace = True)
    print(f"測試集預測的正類數量: {predict['label'].sum()}")

    return predict

def save_prediction(predict, save_path, file_name):
    '''
    儲存預測結果。

    Args:
        predict (pd.DataFrame): 測試集預測結果。
        save_path (str): 儲存路徑。
        file_name (str): 檔案名稱。
    '''
    predict.to_csv(save_path + file_name, index = False) 
