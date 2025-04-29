import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import gc
import warnings
#%%
# CONFIGURATIONS
#显示所有列
pd.set_option('display.max_columns', None)
#数据保留到小数点后两位
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')
#%%
# Import datasets
train = pd.read_csv("../datasets/train.csv")
stores = pd.read_csv("../datasets/stores.csv")
transactions = pd.read_csv("../datasets/transactions.csv").sort_values(["store_nbr", "date"])
oil = pd.read_csv("../datasets/oil.csv")
holidays = pd.read_csv("../datasets/holidays_events.csv")

print("The train data \n")
print(train.head())

print("The stores data \n")
print(stores.head())

print("The transactions data \n")
print(transactions.head())

print("The oil price data \n")
print(oil.head())
#%%
# Datetime
train["date"] = pd.to_datetime(train.date)
transactions["date"] = pd.to_datetime(transactions.date)
oil["date"] = pd.to_datetime(oil.date)
holidays["date"] = pd.to_datetime(holidays.date)

# Data types
train.onpromotion = train.onpromotion.astype("float16")
train.sales = train.sales.astype("float32")
stores.cluster = stores.cluster.astype("int8")
#%%
# Correlation between sales and transactions
#计算train的sales和transactions中的交易量transactions列有多大的关系
#计算 "date" 和 "store_nbr" 分组后的销售总额与 "transactions" 列之间的 Spearman 秩相关系数，并以格式化的方式输出
temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how = "left")
print("Spearman Correlation between Total Sales and Transactions: {:,.4f}".format(temp.corr("spearman").sales.loc["transactions"]))
#%%
#创建了一个新的 DataFrame a，其中包含了按月重新采样后的平均交易数量，并额外添加了一个表示年份的列 "year"。这样的数据整理通常有助于进行时间序列分析或可视化。
a = transactions.set_index("date").resample("M").transactions.mean().reset_index()
a["year"] = a.date.dt.year

for year, year_data in a.groupby("year"):
    plt.plot(year_data["date"], year_data["transactions"], label=f'Year {year}')
plt.title("Monthly Average Transactions")
plt.xlabel("Date")
plt.ylabel("Transactions")
plt.legend()
plt.show()
#%%
a = transactions.copy()
a["year"] = a.date.dt.year
a["dayofweek"] = a.date.dt.dayofweek + 1
a = a.groupby(["year", "dayofweek"]).transactions.mean().reset_index()

for year, year_data in a.groupby("year"):
    plt.plot(year_data["dayofweek"], year_data["transactions"], label=f'Year {year}')
plt.title("Day of Week Transactions")
plt.xlabel("Day of Week")
plt.ylabel("Transactions")
plt.legend()
plt.show()
#%%
# Oil
oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
#将 "dcoilwtico" 列中为 0 的值替换为 NaN。
oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
#将NaN进行插值处理
oil["dcoilwtico_interpolated"] =oil.dcoilwtico.interpolate()
oil['dcoilwtico_interpolated'][0]= oil['dcoilwtico_interpolated'][1]
print(oil.head(40))

fig, ax = plt.subplots(figsize=(10, 6))

oil_melted = oil.melt(id_vars=['date'] + list(oil.keys()[5:]), var_name='Legend')
oil_sorted = oil_melted.sort_values(["Legend", "date"], ascending=[False, True])

for key, grp in oil_sorted.groupby('Legend'):
    ax.plot(grp['date'], grp['value'], label=key)

ax.set_title("Daily Oil Price")
ax.legend()
plt.show()
#%%
# Transactions
daily_transactions = transactions.groupby('date')['transactions'].sum().reset_index()
daily_transactions = daily_transactions.rename(columns={'transactions': 'daily_transactions'})
daily_transactions = daily_transactions.set_index("date").daily_transactions.resample("D").sum().reset_index()
# 将 "daily_transactions" 列中小于等于 25000 的值替换为 NaN。
daily_transactions["daily_transactions"] = np.where(daily_transactions["daily_transactions"] <= 25000, np.nan, daily_transactions["daily_transactions"])
# 在 DataFrame 中添加一个名为 "daily_transactions_interpolated" 的新列，其中包含对 "daily_transactions" 列进行线性插值处理后的值。插值可以用来填补缺失值，使数据更加连续。
daily_transactions["daily_transactions_interpolated"] =daily_transactions.daily_transactions.interpolate()
daily_transactions['daily_transactions_interpolated'][0]= daily_transactions['daily_transactions_interpolated'][1]
print(daily_transactions.head(40))

plt.figure(figsize=(20, 6))
plt.plot(daily_transactions["date"], daily_transactions["daily_transactions_interpolated"], marker='o', linestyle='-')
plt.title('Transactions per_store')
plt.xlabel('Date')
plt.ylabel('Somme des Transactions')
plt.show()
#%%
# Sales
daily_sales = train.groupby('date')['sales'].sum().reset_index()
daily_sales = daily_sales.rename(columns={'sales': 'daily_sales'})
daily_sales = daily_sales.set_index("date").daily_sales.resample("D").sum().reset_index()
daily_sales["daily_sales"] = np.where(daily_sales["daily_sales"] <= 200000, np.nan, daily_sales["daily_sales"])
daily_sales["daily_sales_interpolated"] =daily_sales.daily_sales.interpolate()
daily_sales['daily_sales_interpolated'][0]= daily_sales['daily_sales_interpolated'][1]
print(daily_sales.head(30))

plt.figure(figsize=(20, 6))
plt.plot(daily_sales["date"], daily_sales["daily_sales_interpolated"], marker='o', linestyle='-')
plt.title('daily_sales per_store')
plt.xlabel('Date')
plt.ylabel('daily_sales')
plt.show()
#%%
# Onpromotion
daily_onpromotion = train.groupby('date')['onpromotion'].sum().reset_index()
daily_onpromotion = daily_onpromotion.rename(columns={'onpromotion': 'daily_onpromotion'})
daily_onpromotion = daily_onpromotion.set_index("date").daily_onpromotion.resample("D").sum().reset_index()
daily_onpromotion["daily_onpromotion"] = np.where(daily_onpromotion["daily_onpromotion"] == 0, np.nan, daily_onpromotion["daily_onpromotion"])
#处理 "daily_onpromotion" 列中的缺失值，将特定日期的值设置为 NaN，进行插值处理，并在特定日期范围内将插值后的 NaN 值填充为 0。这可能是为了处理异常值或确保数据在特定日期的连续性。
january_first_2013 = daily_onpromotion[daily_onpromotion['date'] == '2013-01-01'].index[0]
january_first_2014 = daily_onpromotion[daily_onpromotion['date'] == '2014-01-01'].index[0]
january_first_2015 = daily_onpromotion[daily_onpromotion['date'] == '2015-01-01'].index[0]
january_first_2016 = daily_onpromotion[daily_onpromotion['date'] == '2016-01-01'].index[0]
january_first_2017 = daily_onpromotion[daily_onpromotion['date'] == '2017-01-01'].index[0]
daily_onpromotion["daily_onpromotion"][january_first_2013] = np.nan
daily_onpromotion["daily_onpromotion"][january_first_2014] = np.nan
daily_onpromotion["daily_onpromotion"][january_first_2015] = np.nan
daily_onpromotion["daily_onpromotion"][january_first_2016] = np.nan
daily_onpromotion["daily_onpromotion"][january_first_2017] = np.nan

daily_onpromotion["daily_onpromotion_interpolated"] = daily_onpromotion.daily_onpromotion.interpolate()

start_date = '2013-01-01'
end_date = '2014-03-31'
mask = (daily_onpromotion['date'] >= start_date) & (daily_onpromotion['date'] <= end_date)
daily_onpromotion.loc[mask, 'daily_onpromotion_interpolated'] = daily_onpromotion.loc[mask, 'daily_onpromotion_interpolated'].fillna(0)
print(daily_onpromotion.tail(30))

plt.figure(figsize=(20, 6))
plt.plot(daily_onpromotion["date"], daily_onpromotion["daily_onpromotion_interpolated"], marker='o', linestyle='-')
plt.title('daily_onpromotion per_store')
plt.xlabel('Date')
plt.ylabel('Somme des daily_onpromotion')
plt.show()
#%%
# Holidays
#从假期数据中提取国家级假期，将这些假期标记为 1，然后按日重新采样，计算每日国家级假期的数量。最终的结果是一个包含日期和相应国家级假期数量的 DataFrame。
national_holidays = holidays[(holidays['locale'] == 'National') & (holidays['date'] >= '2013-01-01')& (holidays['date'] <= '2017-08-15')]
national_holidays['is_national_holidays'] = 1
is_national_holidays = national_holidays[['date', 'is_national_holidays']].set_index("date").is_national_holidays.resample("D").sum().reset_index()
print(is_national_holidays.head(30))
#%%
# The overall dataset
final_dataset = pd.merge(daily_sales[['date', 'daily_sales_interpolated']], daily_transactions[['date', 'daily_transactions_interpolated']], on='date', how='inner')
final_dataset = pd.merge(final_dataset, daily_onpromotion[['date', 'daily_onpromotion_interpolated']], on='date', how='inner')
final_dataset = pd.merge(final_dataset, oil[['date', 'dcoilwtico_interpolated']], on='date', how='inner')
final_dataset = pd.merge(final_dataset, is_national_holidays, on='date', how='inner')
final_dataset['month'] = final_dataset['date'].dt.month
final_dataset['day_of_week'] = final_dataset['date'].dt.dayofweek + 1

print(final_dataset.head(30))