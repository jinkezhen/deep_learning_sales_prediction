 # Import libraries
import numpy as np
import pandas as pd
import warnings
# CONFIGURATIONS
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')

# Import datasets
train = pd.read_csv("../datasets/train.csv")
stores = pd.read_csv("../datasets/stores.csv")
transactions = pd.read_csv("./datasets/transactions.csv").sort_values(["store_nbr", "date"])
oil = pd.read_csv("../datasets/oil.csv")
holidays = pd.read_csv("../datasets/holidays_events.csv")

# Datetime
train["date"] = pd.to_datetime(train.date)
transactions["date"] = pd.to_datetime(transactions.date)
oil["date"] = pd.to_datetime(oil.date)
holidays["date"] = pd.to_datetime(holidays.date)

# Data types
train.onpromotion = train.onpromotion.astype("float16")
train.sales = train.sales.astype("float32")
stores.cluster = stores.cluster.astype("int8")

# Oil
oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
oil["dcoilwtico_interpolated"] =oil.dcoilwtico.interpolate()
oil['dcoilwtico_interpolated'][0]= oil['dcoilwtico_interpolated'][1]
print(oil.head())

# Sales
daily_sales = train.groupby(['date', 'store_nbr'])['sales'].sum().reset_index()
daily_sales = daily_sales.rename(columns={'sales': 'daily_sales'})
print(daily_sales.head())
# Transactions
daily_transactions = transactions.groupby(['date', 'store_nbr'])['transactions'].sum().reset_index()
daily_transactions = daily_transactions.rename(columns={'transactions': 'daily_transactions'})
print(daily_transactions.head())
# Onpromotion
daily_onpromotion = train.groupby(['date', 'store_nbr'])['onpromotion'].sum().reset_index()
daily_onpromotion = daily_onpromotion.rename(columns={'onpromotion': 'daily_onpromotion'})
print(daily_onpromotion.head())
# National / Regional / Local Holidays
#从 holidays DataFrame 中筛选出 "locale" 为 'National'（国家级）的假期，并在指定日期范围内进行选择。
#添加了一个名为 "is_national_holidays" 的新列，将其值设置为 1，表示这些日期是国家级假期。
#使用 resample("D").sum() 将数据重新采样为每日频率，并计算每日国家级假期的数量。
national_holidays = holidays[(holidays['locale'] == 'National') & (holidays['date'] >= '2013-01-01')& (holidays['date'] <= '2017-08-15')]
national_holidays['is_national_holidays'] = 1
is_national_holidays = national_holidays[['date', 'is_national_holidays']].set_index("date").is_national_holidays.resample("D").sum().reset_index()

regional_holidays = holidays[(holidays['locale'] == 'Regional') & (holidays['date'] >= '2013-01-01')& (holidays['date'] <= '2017-08-15')]
local_holidays = holidays[(holidays['locale'] == 'Local') & (holidays['date'] >= '2013-01-01')& (holidays['date'] <= '2017-08-15')]
results_regional_holidays = pd.merge(regional_holidays, stores, left_on='locale_name', right_on='state', how='inner')
results_local_holidays = pd.merge(local_holidays, stores, left_on='locale_name', right_on='city', how='inner')
stores_regional_holidays = results_regional_holidays[['date', 'store_nbr']]
stores_local_holidays = results_local_holidays[['date', 'store_nbr']]

# The overall dataset
#这一系列的 merge 操作将包含不同信息的 DataFrame（daily_sales、daily_transactions、daily_onpromotion、oi）按照相同的商店编号和日期进行内连接，构成一个包含多种信息的最终数据集 final_dataset。
final_dataset = pd.merge(daily_sales, daily_transactions, on=['store_nbr', 'date'], how='inner')
final_dataset = pd.merge(final_dataset, daily_onpromotion, on=['store_nbr', 'date'], how='inner')
final_dataset = pd.merge(final_dataset, oil[['date', 'dcoilwtico_interpolated']], on='date', how='inner')
final_dataset['day_of_week'] = final_dataset['date'].dt.dayofweek + 1
final_dataset['day_of_month'] = final_dataset['date'].dt.day
final_dataset['month'] = final_dataset['date'].dt.month
final_dataset['year'] = final_dataset['date'].dt.year
final_dataset = pd.merge(final_dataset, is_national_holidays, on='date', how='inner')
final_dataset['is_regional_holidays'] = 0
final_dataset['is_local_holidays'] = 0
#在 final_dataset 中添加两列，分别是 "is_regional_holidays" 和 "is_local_holidays"，用于标记区域级和本地级的假期
for index, row in stores_regional_holidays.iterrows():
    date = row['date']
    nbr = row['store_nbr']
    final_dataset.loc[(final_dataset['date'] == date) & (final_dataset['store_nbr'] == nbr), 'is_regional_holidays'] = 1
for index, row in stores_local_holidays.iterrows():
    date = row['date']
    nbr = row['store_nbr']
    final_dataset.loc[(final_dataset['date'] == date) & (final_dataset['store_nbr'] == nbr), 'is_local_holidays'] = 1

final_dataset.to_csv("final_dataset.csv", index=False)
print(final_dataset.head(30))
#%%
  # Import libraries
import numpy as np
import pandas as pd
import warnings
# CONFIGURATIONS
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')

# Import datasets
train = pd.read_csv("../datasets/train.csv")
stores = pd.read_csv("../datasets/stores.csv")
transactions = pd.read_csv("../datasets/transactions.csv").sort_values(["store_nbr", "date"])
oil = pd.read_csv("../datasets/oil.csv")
holidays = pd.read_csv("../datasets/holidays_events.csv")

# Datetime
train["date"] = pd.to_datetime(train.date)
transactions["date"] = pd.to_datetime(transactions.date)
oil["date"] = pd.to_datetime(oil.date)
holidays["date"] = pd.to_datetime(holidays.date)

# Data types
train.onpromotion = train.onpromotion.astype("float16")
train.sales = train.sales.astype("float32")
stores.cluster = stores.cluster.astype("int8")

# Oil
oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
oil["dcoilwtico_interpolated"] =oil.dcoilwtico.interpolate()
oil['dcoilwtico_interpolated'][0]= oil['dcoilwtico_interpolated'][1]

# Sales
daily_sales = train.groupby(['date', 'store_nbr'])['sales'].sum().reset_index()
daily_sales = daily_sales.rename(columns={'sales': 'daily_sales'})

# Transactions
daily_transactions = transactions.groupby(['date', 'store_nbr'])['transactions'].sum().reset_index()
daily_transactions = daily_transactions.rename(columns={'transactions': 'daily_transactions'})

# Onpromotion
daily_onpromotion = train.groupby(['date', 'store_nbr'])['onpromotion'].sum().reset_index()
daily_onpromotion = daily_onpromotion.rename(columns={'onpromotion': 'daily_onpromotion'})

# National / Regional / Local Holidays
#从 holidays DataFrame 中筛选出 "locale" 为 'National'（国家级）的假期，并在指定日期范围内进行选择。
#添加了一个名为 "is_national_holidays" 的新列，将其值设置为 1，表示这些日期是国家级假期。
#使用 resample("D").sum() 将数据重新采样为每日频率，并计算每日国家级假期的数量。
national_holidays = holidays[(holidays['locale'] == 'National') & (holidays['date'] >= '2013-01-01')& (holidays['date'] <= '2017-08-15')]
national_holidays['is_national_holidays'] = 1
is_national_holidays = national_holidays[['date', 'is_national_holidays']].set_index("date").is_national_holidays.resample("D").sum().reset_index()

regional_holidays = holidays[(holidays['locale'] == 'Regional') & (holidays['date'] >= '2013-01-01')& (holidays['date'] <= '2017-08-15')]
local_holidays = holidays[(holidays['locale'] == 'Local') & (holidays['date'] >= '2013-01-01')& (holidays['date'] <= '2017-08-15')]
results_regional_holidays = pd.merge(regional_holidays, stores, left_on='locale_name', right_on='state', how='inner')
results_local_holidays = pd.merge(local_holidays, stores, left_on='locale_name', right_on='city', how='inner')
stores_regional_holidays = results_regional_holidays[['date', 'store_nbr']]
stores_local_holidays = results_local_holidays[['date', 'store_nbr']]

# The overall dataset
#这一系列的 merge 操作将包含不同信息的 DataFrame（daily_sales、daily_transactions、daily_onpromotion、oi）按照相同的商店编号和日期进行内连接，构成一个包含多种信息的最终数据集 final_dataset。
final_dataset = pd.merge(daily_sales, daily_transactions, on=['store_nbr', 'date'], how='inner')
final_dataset = pd.merge(final_dataset, daily_onpromotion, on=['store_nbr', 'date'], how='inner')
final_dataset = pd.merge(final_dataset, oil[['date', 'dcoilwtico_interpolated']], on='date', how='inner')
final_dataset['day_of_week'] = final_dataset['date'].dt.dayofweek + 1
final_dataset['day_of_month'] = final_dataset['date'].dt.day
final_dataset['month'] = final_dataset['date'].dt.month
final_dataset['year'] = final_dataset['date'].dt.year
final_dataset = pd.merge(final_dataset, is_national_holidays, on='date', how='inner')
final_dataset['is_regional_holidays'] = 0
final_dataset['is_local_holidays'] = 0
#在 final_dataset 中添加两列，分别是 "is_regional_holidays" 和 "is_local_holidays"，用于标记区域级和本地级的假期
for index, row in stores_regional_holidays.iterrows():
    date = row['date']
    nbr = row['store_nbr']
    final_dataset.loc[(final_dataset['date'] == date) & (final_dataset['store_nbr'] == nbr), 'is_regional_holidays'] = 1
for index, row in stores_local_holidays.iterrows():
    date = row['date']
    nbr = row['store_nbr']
    final_dataset.loc[(final_dataset['date'] == date) & (final_dataset['store_nbr'] == nbr), 'is_local_holidays'] = 1

final_dataset.to_csv("final_dataset.csv", index=False)