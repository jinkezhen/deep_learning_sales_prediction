# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import gc
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# %%
# CONFIGURATIONS
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')

# Import datasets
dataset = pd.read_csv("../datasets/final_dataset.csv")
print(dataset.head())
dataset = dataset.drop(dataset[(dataset['month'] == 1) & (dataset['day_of_month'] == 1)].index)
# %%
# Normalize the data
# 标准化指定列的数据
selected_columns = ['daily_sales', 'daily_transactions', 'daily_onpromotion', 'dcoilwtico_interpolated']
dataset_selected = dataset[selected_columns]
daily_sales_mean = dataset['daily_sales'].mean()
daily_sales_std = dataset['daily_sales'].std()
# 对选择的列进行标准化，即将每个值减去均值，然后除以标准差。这样可以使得数据在不同列之间具有相同的尺度，有助于某些模型的训练和预测。
normalized_dataset = (dataset_selected - dataset_selected.mean()) / dataset_selected.std()
dataset[selected_columns] = normalized_dataset
dataset['year'] = dataset['year'] - 2012

# Split the data
train = dataset.iloc[:65129]
test = dataset.iloc[65129:]


# %%
def make_store_prediction_LSTM(store_nb, lmst_nb):
    # Prepare the data
    # 选择 train 数据集中 "store_nbr" 列等于给定 store_nb 值的行，即筛选出指定商店的数据。
    # 删除 "store_nbr" 列，因为已经选择了特定商店。
    # 将 "date" 列转换为 datetime 类型，可能是为了在绘图轴上使用日期。
    # 删除 "date" 列，因为已经将日期转换为 datetime 类型。
    # 重新设置索引，丢弃原有索引，并在原地进行修改。
    train_data = train[train['store_nbr'] == store_nb]
    train_data = train_data.drop('store_nbr', axis=1)
    date_train = pd.to_datetime(train_data['date'])  # usefull the the date on the plot axis
    train_data = train_data.drop("date", axis=1)
    train_data.reset_index(drop=True, inplace=True)
    test_data = test[test['store_nbr'] == store_nb]
    test_data = test_data.drop('store_nbr', axis=1)
    date_test = pd.to_datetime(test_data['date'])  # usefull the the date on the plot axis
    test_data = test_data.drop("date", axis=1)
    test_data.reset_index(drop=True, inplace=True)

    # Create the X_train, X_test, Y_train and Y_test variables
    X_str_train = train_data.drop('daily_sales', axis=1)
    Y_str_train = train_data['daily_sales']
    X_str_test = test_data.drop('daily_sales', axis=1)
    Y_str_test = test_data['daily_sales']

    # Reshape the variables for the LSTM model
    # X_str_train.shape[0]: 表示 X_str_train 中样本的数量。
    # X_str_train.shape[1]: 表示每个样本的时间步数（或特征数量，根据具体问题而定）。
    # X_str_train.shape[2]: 表示每个时间步（或特征）的维度
    X_str_train = X_str_train.to_numpy()
    X_str_train = X_str_train.reshape((X_str_train.shape[0], 1, X_str_train.shape[1]))
    X_str_test = X_str_test.to_numpy()
    X_str_test = X_str_test.reshape((X_str_test.shape[0], 1, X_str_test.shape[1]))

    # Construction of the LSTM NN
    # input shape是3维: (Batch_size, Time_step, Input_Sizes), 其中Time_step是时间序列的长度, 对应到语句里就是语句的最大长度; Input_Sizes是每个时间点输入x的维度, 对于语句来说,就是一个字的embedding的向量维度
    model2 = Sequential()
    model2.add(LSTM(lmst_nb, input_shape=(X_str_train.shape[1], X_str_train.shape[2])))
    model2.add(Dense(1))
    model2.compile(optimizer='adam', loss='mean_squared_error')
    # epochs=20 表示模型将遍历整个训练集 20 次。 batch_size=31 表示每个训练批次的样本数为 31
    model2.fit(X_str_train, Y_str_train, epochs=20, batch_size=31, validation_data=(X_str_test, Y_str_test))
    # model2.fit(X_str_train, Y_str_train, epochs=20, batch_size=32, validation_split=(X_str_train, Y_str_train))

    # Model evaluation
    loss2 = model2.evaluate(X_str_test, Y_str_test)
    print(f'Loss (Mean Squared Error) sur l\'ensemble de test : {loss2}')

    # Prediction on training and test sets
    Y_train_predict = model2.predict(X_str_train)
    Y_test_predict = model2.predict(X_str_test)

    # calculate root mean squared error
    train_rmse = math.sqrt(mean_squared_error(Y_str_train, Y_train_predict[:,0]))
    # RMSE为均方根误差
    print('Train Score: %.2f RMSE' % (train_rmse))
    test_rmse = math.sqrt(mean_squared_error(Y_str_test, Y_test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % (test_rmse))

    # Calculate R2，决定系数
    # r2_score为判定系数介于负无穷-1之间，越大越好
    train_r2 = r2_score(Y_str_train, Y_train_predict)
    print('Train Score: %.2f R2' % (train_r2))
    test_r2 = r2_score(Y_str_test, Y_test_predict)
    print('Test Score: %.2f R2' % (test_r2))

    # Calculate the RMSE and R² std
    rmse_std = np.std([train_rmse, test_rmse])
    r2_std = np.std([train_r2, test_r2])
    print(f'Standard Deviation of RMSE: {rmse_std:.2f}')
    print(f'Standard Deviation of R2: {r2_std:.2f}')

    # Find non-standardized daily_sales
    # 将数据反标准化，进行绘图
    Y_str_train = (Y_str_train * daily_sales_std) + daily_sales_mean
    Y_str_test = (Y_str_test * daily_sales_std) + daily_sales_mean
    Y_train_predict[:, 0] = (Y_train_predict[:, 0] * daily_sales_std) + daily_sales_mean
    Y_test_predict[:, 0] = (Y_test_predict[:, 0] * daily_sales_std) + daily_sales_mean

    # Plot for training set
    plt.figure(figsize=(20, 6))
    plt.subplot(2, 1, 1)
    plt.plot(date_train, Y_str_train, label='Real values')
    plt.plot(date_train, Y_train_predict, label='Predicted values')
    plt.title(f'Comparison between LSTM predicted/real values on the train for the store {store_nb}',fontsize=14)
    plt.legend(fontsize=14)

    # Plot for testing set
    plt.subplot(2, 1, 2)
    plt.plot(date_test, Y_str_test, label='Real values')
    plt.plot(date_test, Y_test_predict, label='Predicted values')
    plt.title(f'Comparison between LSTM predicted/real values on the test for the store {store_nb}',fontsize=14)
    plt.legend(fontsize=14)

    # Plot pour l'ensemble d'entraînement et de test sur un unique graphique
    # plt.plot(date_train, Y_str_train, label='Real values (Train)', color='blue')
    # plt.plot(date_train, Y_train_predict, label='Predicted values (Train & Test)', color='orange')
    # plt.plot(date_test, Y_test_predict, color='orange')
    # plt.title(f'Comparison between predicted/real values on the train and test for the store {store_nb}')
    # plt.xlabel('Month and Year')
    # plt.legend()

    plt.tight_layout()
    plt.show()


# %%
def make_store_prediction_RNN(store_nb, lmst_nb):
    train_data = train[train['store_nbr'] == store_nb]
    train_data = train_data.drop('store_nbr', axis=1)
    date_train = pd.to_datetime(train_data['date'])
    train_data = train_data.drop("date", axis=1)
    train_data.reset_index(drop=True, inplace=True)
    test_data = test[test['store_nbr'] == store_nb]
    test_data = test_data.drop('store_nbr', axis=1)
    date_test = pd.to_datetime(test_data['date'])
    test_data = test_data.drop("date", axis=1)
    test_data.reset_index(drop=True, inplace=True)
    X_str_train = train_data.drop('daily_sales', axis=1)
    Y_str_train = train_data['daily_sales']
    X_str_test = test_data.drop('daily_sales', axis=1)
    Y_str_test = test_data['daily_sales']
    X_str_train = X_str_train.to_numpy()
    X_str_train = X_str_train.reshape((X_str_train.shape[0], 1, X_str_train.shape[1]))
    X_str_test = X_str_test.to_numpy()
    X_str_test = X_str_test.reshape((X_str_test.shape[0], 1, X_str_test.shape[1]))
    model2 = Sequential()
    model2.add(SimpleRNN(lmst_nb, input_shape=(X_str_train.shape[1], X_str_train.shape[2])))
    model2.add(Dense(1))
    model2.compile(optimizer='adam', loss='mean_squared_error')
    model2.fit(X_str_train, Y_str_train, epochs=20, batch_size=31, validation_data=(X_str_test, Y_str_test))

    loss2 = model2.evaluate(X_str_test, Y_str_test)
    print(f'Loss (Mean Squared Error) sur l\'ensemble de test : {loss2}')

    # Prediction on training and test sets
    Y_train_predict = model2.predict(X_str_train)
    Y_test_predict = model2.predict(X_str_test)

    # calculate root mean squared error
    train_rmse = math.sqrt(mean_squared_error(Y_str_train, Y_train_predict[:, 0]))
    print('Train Score: %.2f RMSE' % (train_rmse))
    test_rmse = math.sqrt(mean_squared_error(Y_str_test, Y_test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % (test_rmse))

    # Calculate R2
    train_r2 = r2_score(Y_str_train, Y_train_predict)
    print('Train Score: %.2f R2' % (train_r2))
    test_r2 = r2_score(Y_str_test, Y_test_predict)
    print('Test Score: %.2f R2' % (test_r2))

    # Calculate the RMSE and R² std
    rmse_std = np.std([train_rmse, test_rmse])
    r2_std = np.std([train_r2, test_r2])
    print(f'Standard Deviation of RMSE: {rmse_std:.2f}')
    print(f'Standard Deviation of R2: {r2_std:.2f}')

    # Find non-standardized daily_sales
    Y_str_train = (Y_str_train * daily_sales_std) + daily_sales_mean
    Y_str_test = (Y_str_test * daily_sales_std) + daily_sales_mean
    Y_train_predict[:, 0] = (Y_train_predict[:, 0] * daily_sales_std) + daily_sales_mean
    Y_test_predict[:, 0] = (Y_test_predict[:, 0] * daily_sales_std) + daily_sales_mean

    # Plot for training set
    plt.figure(figsize=(20, 6))
    plt.subplot(2, 1, 1)
    plt.plot(date_train, Y_str_train, label='Real values')
    plt.plot(date_train, Y_train_predict, label='Predicted values')
    plt.title(f'Comparison between Simple RNN predicted/real values on the train for the store {store_nb}',fontsize=14)
    plt.legend(fontsize=14)

    # Plot for testing set
    plt.subplot(2, 1, 2)
    plt.plot(date_test, Y_str_test, label='Real values')
    plt.plot(date_test, Y_test_predict, label='Predicted values')
    plt.title(f'Comparison between Simple RNN predicted/real values on the test for the store {store_nb}',fontsize=14)
    plt.legend(fontsize=14)

    # Plot pour l'ensemble d'entraînement et de test sur un unique graphique
    # plt.plot(date_train, Y_str_train, label='Real values (Train)', color='blue')
    # plt.plot(date_train, Y_train_predict, label='Predicted values (Train & Test)', color='orange')
    # plt.plot(date_test, Y_test_predict, color='orange')
    # plt.title(f'Comparison between predicted/real values on the train and test for the store {store_nb}')
    # plt.xlabel('Month and Year')
    # plt.legend()

    plt.tight_layout()
    plt.show()


# %%
def make_store_prediction_GRU(store_nb, lmst_nb):
    # Prepare the data
    train_data = train[train['store_nbr'] == store_nb]
    train_data = train_data.drop('store_nbr', axis=1)
    date_train = pd.to_datetime(train_data['date'])  # usefull the the date on the plot axis
    train_data = train_data.drop("date", axis=1)
    train_data.reset_index(drop=True, inplace=True)
    test_data = test[test['store_nbr'] == store_nb]
    test_data = test_data.drop('store_nbr', axis=1)
    date_test = pd.to_datetime(test_data['date'])  # usefull the the date on the plot axis
    test_data = test_data.drop("date", axis=1)
    test_data.reset_index(drop=True, inplace=True)

    # Create the X_train, X_test, Y_train and Y_test variables
    X_str_train = train_data.drop('daily_sales', axis=1)
    Y_str_train = train_data['daily_sales']
    X_str_test = test_data.drop('daily_sales', axis=1)
    Y_str_test = test_data['daily_sales']

    # Reshape the variables for the GRU model
    X_str_train = X_str_train.to_numpy()
    X_str_train = X_str_train.reshape((X_str_train.shape[0], 1, X_str_train.shape[1]))
    X_str_test = X_str_test.to_numpy()
    X_str_test = X_str_test.reshape((X_str_test.shape[0], 1, X_str_test.shape[1]))

    # Construction of the GRU NN
    model2 = Sequential()
    model2.add(GRU(lmst_nb, input_shape=(X_str_train.shape[1], X_str_train.shape[2])))
    model2.add(Dense(1))
    model2.compile(optimizer='adam', loss='mean_squared_error')
    model2.fit(X_str_train, Y_str_train, epochs=20, batch_size=31, validation_data=(X_str_test, Y_str_test))

    # Model evaluation
    loss2 = model2.evaluate(X_str_test, Y_str_test)
    print(f'Loss (Mean Squared Error) sur l\'ensemble de test : {loss2}')

    # Prediction on training and test sets
    Y_train_predict = model2.predict(X_str_train)
    Y_test_predict = model2.predict(X_str_test)

    # calculate root mean squared error
    train_rmse = math.sqrt(mean_squared_error(Y_str_train, Y_train_predict[:, 0]))
    print('Train Score: %.2f RMSE' % (train_rmse))
    test_rmse = math.sqrt(mean_squared_error(Y_str_test, Y_test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % (test_rmse))

    # Calculate R2
    train_r2 = r2_score(Y_str_train, Y_train_predict)
    print('Train Score: %.2f R2' % (train_r2))
    test_r2 = r2_score(Y_str_test, Y_test_predict)
    print('Test Score: %.2f R2' % (test_r2))

    # Calculate the RMSE and R² std
    rmse_std = np.std([train_rmse, test_rmse])
    r2_std = np.std([train_r2, test_r2])
    print(f'Standard Deviation of RMSE: {rmse_std:.2f}')
    print(f'Standard Deviation of R2: {r2_std:.2f}')

    # Find non-standardized daily_sales
    Y_str_train = (Y_str_train * daily_sales_std) + daily_sales_mean
    Y_str_test = (Y_str_test * daily_sales_std) + daily_sales_mean
    Y_train_predict[:, 0] = (Y_train_predict[:, 0] * daily_sales_std) + daily_sales_mean
    Y_test_predict[:, 0] = (Y_test_predict[:, 0] * daily_sales_std) + daily_sales_mean

    # Plot for training set
    plt.figure(figsize=(20, 6))
    plt.subplot(2, 1, 1)
    plt.plot(date_train, Y_str_train, label='Real values')
    plt.plot(date_train, Y_train_predict, label='Predicted values')
    plt.title(f'Comparison between GRU predicted/real values on the train for the store {store_nb}',fontsize=14)
    plt.legend(fontsize=14)

    # Plot for testing set
    plt.subplot(2, 1, 2)
    plt.plot(date_test, Y_str_test, label='Real values')
    plt.plot(date_test, Y_test_predict, label='Predicted values')
    plt.title(f'Comparison between GRU predicted/real values on the test for the store {store_nb}',fontsize=14)
    plt.legend(fontsize=14)

    # Plot pour l'ensemble d'entraînement et de test sur un unique graphique
    # plt.plot(date_train, Y_str_train, label='Real values (Train)', color='blue')
    # plt.plot(date_train, Y_train_predict, label='Predicted values (Train & Test)', color='orange')
    # plt.plot(date_test, Y_test_predict, color='orange')
    # plt.title(f'Comparison between predicted/real values on the train and test for the store {store_nb}')
    # plt.xlabel('Month and Year')
    # plt.legend()

    plt.tight_layout()
    plt.show()


# %%
# Make daily sales predictions for the store of your choice and the number of LSTM neurons of your choice

make_store_prediction_LSTM(1, 32)
# Make daily sales predictions for the store of your choice and the number of Simple RNN neurons of your choice

make_store_prediction_RNN(1, 32)
# %%
# Make daily sales predictions for the store of your choice and the number of GRU neurons of your choice

make_store_prediction_GRU(1, 32)