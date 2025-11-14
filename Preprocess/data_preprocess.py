import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

def data_preprocess(acct_transaction, acct_alert, acct_test, currencies_rate):
    '''
    載入、轉換交易資料，並建構圖神經網路所需的 Data 物件。

    本函式執行資料前處理，將原始交易進行歸戶建立節點特徵、邊索引
    切分訓練、驗證、待預測遮罩的 PyTorch Geometric Data 物件。

    Args:
        acct_transaction (pd.DataFrame): 原始的帳戶交易紀錄資料。
        acct_alert (pd.DataFrame): 警示帳戶清單，用於產生標籤。
        acct_test (pd.DataFrame): 待預測帳戶清單。
        currencies_rate (pd.DataFrame): 預先下載好的匯率轉換檔案。

    Returns:
        data: 包含所有節點特徵、邊索引的 Data 物件。
        accounts: 所有帳戶。
        y: 警示帳戶標籤。
    '''
    # 進行幣別的轉換
    rate_map = dict(zip(currencies_rate['currency'], currencies_rate['rate_to_twd']))
    rate_map['USD'] = currencies_rate.loc[currencies_rate['currency'] == 'TWD', 'Exrate'].iloc[0]
    acct_transaction['twd_rate'] = acct_transaction['currency_type'].map(rate_map)
    acct_transaction['final_amt'] = acct_transaction['txn_amt'] * acct_transaction['twd_rate']

    all_accounts = set(acct_transaction['from_acct'].unique()) | set(acct_transaction['to_acct'].unique())
    accounts = sorted(list(all_accounts))
    account_to_idx = {acc: idx for idx, acc in enumerate(accounts)}
    num_nodes = len(accounts)
    yuanshan_from = set(acct_transaction[acct_transaction['from_acct_type'] == 1]['from_acct'].unique())
    yuanshan_to = set(acct_transaction[acct_transaction['to_acct_type'] == 1]['to_acct'].unique())
    yuanshan_accounts = yuanshan_from | yuanshan_to
    alert_set = set(acct_alert['acct'].values)
    test_accounts = set(acct_test['acct'].values)
    
    # 各帳戶分別做為轉出或匯入帳戶建立特徵
    from_agg = acct_transaction.groupby('from_acct').agg({
        'final_amt' : ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'from_acct_type' : lambda x : 1 if len(x) > 0 and x.mode()[0] == 1 else 0,
        'is_self_txn' : lambda x : (x == 'Y').sum(),
        'txn_time' : 'nunique',
        'channel_type' : 'nunique',  
        'to_acct' : 'nunique'}).fillna(0)

    from_agg.columns = ['from_count', 'from_sum', 'from_mean', 'from_std', 'from_min', 'from_max', 
                        'from_acct_type', 'self_txn', 'time_diversity', 'channel_diversity', 'counterparty_diversity']

    to_agg = acct_transaction.groupby('to_acct').agg({
        'final_amt' : ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'to_acct_type' : lambda x : 1 if len(x) > 0 and x.mode()[0] == 1 else 0,
        'is_self_txn' : lambda x : (x == 'Y').sum(),
        'txn_time' : 'nunique',
        'channel_type' : 'nunique',
        'from_acct' : 'nunique'}).fillna(0)

    to_agg.columns = ['to_count', 'to_sum', 'to_mean', 'to_std', 'to_min', 'to_max', 'to_acct_type',
                    'to_self_txn', 'to_time_diversity', 'to_channel_diversity', 'payer_diversity']

    from_agg['from_amt_cv'] = from_agg['from_std'] / (from_agg['from_mean'] + 1e-6)  
    from_agg['from_self_txn_ratio'] = from_agg['self_txn'] / (from_agg['from_count'] + 1e-6)
    from_agg['from_avg_counterparty_txn'] = from_agg['from_count'] / (from_agg['counterparty_diversity'] + 1e-6) 
    to_agg['to_amt_cv'] = to_agg['to_std'] / (to_agg['to_mean'] + 1e-6) 
    to_agg['to_self_txn_ratio'] = to_agg['to_self_txn'] / (to_agg['to_count'] + 1e-6)
    to_agg['to_avg_payer_txn'] = to_agg['to_count'] / (to_agg['payer_diversity'] + 1e-6)

    node_features = []
    node_labels = {}

    for idx, acct in enumerate(accounts):
        features = []
        if acct in from_agg.index:
            features.extend([
                from_agg.loc[acct, 'from_count'],
                from_agg.loc[acct, 'from_sum'],
                from_agg.loc[acct, 'from_mean'],
                from_agg.loc[acct, 'from_std'],
                from_agg.loc[acct, 'from_min'],
                from_agg.loc[acct, 'from_max'],
                from_agg.loc[acct, 'from_acct_type'],
                from_agg.loc[acct, 'self_txn'],
                from_agg.loc[acct, 'time_diversity'],
                from_agg.loc[acct, 'channel_diversity'],
                from_agg.loc[acct, 'counterparty_diversity'],
                from_agg.loc[acct, 'from_amt_cv'],
                from_agg.loc[acct, 'from_self_txn_ratio'],
                from_agg.loc[acct, 'from_avg_counterparty_txn']])
        else:
            features.extend([0] * 14)

        if acct in to_agg.index:
            features.extend([
                to_agg.loc[acct, 'to_count'],
                to_agg.loc[acct, 'to_sum'],
                to_agg.loc[acct, 'to_mean'],
                to_agg.loc[acct, 'to_std'],
                to_agg.loc[acct, 'to_min'],
                to_agg.loc[acct, 'to_max'],
                to_agg.loc[acct, 'to_acct_type'],
                to_agg.loc[acct, 'to_self_txn'],
                to_agg.loc[acct, 'to_time_diversity'],
                to_agg.loc[acct, 'to_channel_diversity'],
                to_agg.loc[acct, 'payer_diversity'],
                to_agg.loc[acct, 'to_amt_cv'],
                to_agg.loc[acct, 'to_self_txn_ratio'],           
                to_agg.loc[acct, 'to_avg_payer_txn']
                ])
        else:
            features.extend([0] * 14)
        
        node_features.append(features)
        node_labels[idx] = 1 if acct in alert_set else 0

    node_features = np.array(node_features, dtype = np.float32)
    y = torch.tensor([node_labels[i] for i in range(num_nodes)], dtype = torch.long)

    # 建立圖神經網路所需的邊索引，並建立成無向圖
    edges_set = {
        (account_to_idx[from_acct], account_to_idx[to_acct])
        for from_acct, to_acct in zip(acct_transaction['from_acct'], acct_transaction['to_acct'])}
    edge_list = list(edges_set)
    edge_index = torch.tensor(edge_list, dtype = torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    # 切分出訓練、驗證及測試集，並進行標準化
    train_val_indices = []
    test_indices = []

    for idx, acct in enumerate(accounts):
        if acct in yuanshan_accounts:
            if acct in test_accounts:
                test_indices.append(idx)
            else:
                train_val_indices.append(idx)
    train_val_labels = y[train_val_indices]

    train_indices_list, val_indices_list = train_test_split(
        train_val_indices,
        test_size = 0.1,
        random_state = 42,
        stratify = train_val_labels)

    train_mask = torch.zeros(num_nodes, dtype = torch.bool)
    val_mask = torch.zeros(num_nodes, dtype = torch.bool)
    test_mask = torch.zeros(num_nodes, dtype = torch.bool)

    for idx in train_indices_list:
        train_mask[idx] = True
    for idx in val_indices_list:
        val_mask[idx] = True
    for idx in test_indices:
        test_mask[idx] = True

    scaler = StandardScaler()
    node_features_scaled = node_features.copy()
    train_features = node_features[train_mask]
    scaler.fit(train_features)
    node_features_scaled = scaler.transform(node_features)
    node_features_scaled = node_features_scaled.astype(np.float32)

    x = torch.tensor(node_features_scaled, dtype = torch.float32)

    data = Data(
        x = x,
        edge_index = edge_index,
        y = y,
        train_mask = train_mask,
        val_mask = val_mask,
        test_mask = test_mask)

    return data, accounts, y