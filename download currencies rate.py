import pandas as pd
import requests

acct_transaction = pd.read_csv('/mnt/d/AI CUP/DATA/初賽資料/acct_transaction.csv')
currency_list = acct_transaction['currency_type'].unique().tolist()
currency_list.remove('USD')
response = requests.get('https://tw.rter.info/capi.php')
currency_rate = response.json()
rate_list = []
for i in currency_list:
    rate_list.append(currency_rate[f'USD{i}'])
currencies_rate = pd.DataFrame(rate_list)
currencies_rate['currency'] = currency_list
USDTWD_rate = currencies_rate.loc[currencies_rate['currency'] == 'TWD', 'Exrate'].iloc[0]
currencies_rate['rate_to_twd'] = USDTWD_rate / currencies_rate['Exrate']