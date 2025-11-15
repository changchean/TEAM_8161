#   AI CUP 2025 玉山人工智慧公開挑戰賽－AI偵探出任務，精準揪出警示帳戶！
##  主要功能
- 讀取交易流水帳資料、警示帳戶資料、待預測帳戶資料
- 資料前處理，每個帳戶進行歸戶建立特徵，並且建立後續使用之模型所需項目
- 切分出訓練、驗證、待預測資料集
- 使用 torch_geometric 的圖神經網路進行訓練與預測
## 環境配置
- 查看 requirements.txt
## 使用方式
1.安裝套件
- python 3.11 或以上以及以下套件
  - pandas == 2.3.0
  - requests == 2.32.4
  - numpy == 2.3.3
  - torch == 2.7.1+cu128
  - torch_geometric == 2.7.0
  - scikit-learn == 1.7.0
- 安裝指令

        pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
        pip install pandas==2.3.0 requests numpy==2.3.3 torch_geometric scikit-learn==1.7.0
2.檔案說明
- download currencies rate.py
  - 下載貨幣匯率，並進行各貨幣對台幣的匯率
- Preprocess 資料集中 data_preprocess.py
  - 資料前處理，建立節點特徵、邊索引，包裝成圖神經網路所需的資料格式
- Model 資料夾中 modeling.py
  - 模型訓練、驗證及預測
- main.py
  - 執行包含所有 data_preprocess.py 及 modeling.py 完整過程
- inference.py
  - 單存載入前處理好資料及訓練好的模型做預測
- currency_rate.pkl
  - 下載的匯率表
- preprocess data.7z 壓縮檔中 preprocess data.pt
  - 資料前處理後，包裝成圖神經網路所需的資料格式
- final model.pt
  - 最後一次上傳成績的模型

3.資料準備

資料路徑若不同，可於 main.py 、 inference.py 中修改
- 執行整個流程
  - acct_transaction.csv
  - acct_alert.csv
  - acct_predict.csv
  - currency_rate.pkl
- 使用已訓練好的模型進行預測
  - preprocess data.pt
  - final model.pt
  - acct_predict.csv

4.執行程式

py 檔需在同一路徑下執行，使用之路徑可於 main.py 、 inference.py 中 os.chdir 修改
- 整個流程   
        
        python main.py
- 使用已訓練好的模型進行預測   

        python inference.py
  - 預測結果 predict ，可於 save_path 修改儲存路徑， file_name 修改檔名
## data_preprocessing.py
### 主要函式
- mapping_currencies(): 轉換幣別金額。
- establish_features(): 建立帳戶清單、節點特徵及邊索引。
- train_val_test_split(): 切分出訓練、驗證及測試的遮罩。
- stand_scale(): 特徵資料進行標準化。
- pack_data(): 資料包裝成圖神經網路所需之格式
## modeling.py
### 主要函式
- load_data(): 載入前處理好的資料
- set_seed(): 設定隨機種子
- PULoss(): Positive-Unlabeled (PU) Learning 概念損失函式
- GraphSAGE(): GraphSAGE 模型，並做加權的特徵融合
- training_setups(): 設定損失函數、優化器等
- train_model(): 訓練迭代
- validate_model(): 驗證模型
- training_loop(): 完整訓練循環
- find_best_threshold(): 使用驗證集尋找最佳閾值的 F1
- evaluate_model(): 使用最佳閾值對訓練集、驗證集及測試集進行預測計算 F1
- save_prediction(): 儲存預測結果
## 結果
- 最佳閾值: 0.95
- Train F1 Score: 0.6862
- Val F1 Score: 0.5842
- 上傳 Public F1: 0.784
- 上傳 Private F1: 0.722222
