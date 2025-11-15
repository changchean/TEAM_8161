import os
original_dir = os.getcwd()
os.chdir('/mnt/d/AI CUP/Upload/')
from data_preprocess import *
from modeling import *

if __name__ == '__main__':
    acct_transaction = pd.read_csv('/mnt/d/AI CUP/DATA/初賽資料/acct_transaction.csv')
    acct_alert = pd.read_csv('/mnt/d/AI CUP/DATA/初賽資料/acct_alert.csv')
    acct_test = pd.read_csv('/mnt/d/AI CUP/DATA/初賽資料/acct_predict.csv')
    currencies_rate = pd.read_pickle('/mnt/d/AI CUP/Upload/currency_rate.pkl')
    save_path = '/mnt/d/AI CUP/Upload/'
    file_name = 'final_predict.csv'
    acct_transaction = mapping_currencies(acct_transaction, currencies_rate)
    accounts, yuanshan_accounts, test_accounts, node_features, y, num_nodes, edge_index = establish_features(
        acct_transaction, acct_alert, acct_test)
    train_mask, val_mask, test_mask = train_val_test_split(accounts, yuanshan_accounts, test_accounts, y, num_nodes)
    node_features_scaled = stand_scale(node_features, train_mask)
    data = pack_data(node_features_scaled, edge_index, y, train_mask, val_mask, test_mask)
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(
        in_channels = data.x.shape[1], hidden_channels = 56, out_channels = 2, 
        num_layers = 4, aggr = 'mean', dropout = 0.3).to(device)
    criterion, optimizer, compiled_model, scaler = training_setups(model, data, device)
    data = data.to(device)
    training_loop(compiled_model, model, data, criterion, optimizer, scaler, device)
    model.load_state_dict(torch.load('best_model.pt'))
    predict = evaluate_model(model, data, accounts, acct_test)
    save_prediction(predict, save_path, file_name)
    os.chdir(original_dir)
