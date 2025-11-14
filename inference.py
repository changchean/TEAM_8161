import os
original_dir = os.getcwd()
os.chdir('/mnt/d/AI CUP/Upload/')
from modeling import *

if __name__ == '__main__':
    data_path = '/mnt/d/AI CUP/Upload/preprocess data.pt'
    predict_path = '/mnt/d/AI CUP/DATA/初賽資料/acct_predict.csv'
    model_path = '/mnt/d/AI CUP/Upload/final model.pt'
    save_path = '/mnt/d/AI CUP/Upload/'
    file_name = 'final_predict.csv'
    data, accounts, y, acct_test = load_data(data_path, predict_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = GraphSAGE(
        in_channels = data.x.shape[1], hidden_channels = 56, out_channels = 2, 
        num_layers = 4, aggr = 'mean', dropout = 0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location = device))
    predict = evaluate_model(model, data, accounts, acct_test)
    save_prediction(predict, save_path, file_name)
    os.chdir(original_dir)