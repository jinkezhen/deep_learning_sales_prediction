import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
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
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
content=pd.read_csv("../datasets/bicycle_dataset.csv")
print(content.head())
print(len(content))
content["dteday"]=pd.to_datetime(content["dteday"],format='%d-%m-%Y')
# content['day_of_week'] = content['dteday'].dt.dayofweek + 1
# content['day_of_month'] = content['dteday'].dt.day
# content['month'] = content['dteday'].dt.month
# content['year'] = content['dteday'].dt.year
date_for_plt=content["dteday"]
date_train=date_for_plt[:650]
date_test=date_for_plt[650:]

selected_columns = ['temp','atemp','hum','windspeed','casual','registered','cnt']
dataset_selected =content[selected_columns]
daily_sales_mean = content["cnt"].mean()
daily_sales_std = content["cnt"].std()
normalized_dataset = (dataset_selected - dataset_selected.mean()) / dataset_selected.std()
content[selected_columns] = normalized_dataset

content=content.drop(['dteday'],axis=1)
train=content[:650]
test=content[650:]
x_train=train.drop(['cnt'],axis=1)
y_train=train['cnt']
x_test=test.drop(['cnt'],axis=1)
y_test=test['cnt']

def make_store_prediction_LSTM(lmst_nb):
    # Reshape the variables for the LSTM model
    # X_str_train.shape[0]: 表示 X_str_train 中样本的数量。
    # X_str_train.shape[1]: 表示每个样本的时间步数（或特征数量，根据具体问题而定）。
    # X_str_train.shape[2]: 表示每个时间步（或特征）的维度
    X_str_train =x_train.to_numpy()
    X_str_train = X_str_train.reshape((X_str_train.shape[0], 1, X_str_train.shape[1]))
    X_str_test = x_test.to_numpy()
    X_str_test = X_str_test.reshape((X_str_test.shape[0], 1, X_str_test.shape[1]))

    # Construction of the LSTM NN
    # input shape是3维: (Batch_size, Time_step, Input_Sizes), 其中Time_step是时间序列的长度, 对应到语句里就是语句的最大长度; Input_Sizes是每个时间点输入x的维度, 对于语句来说,就是一个字的embedding的向量维度
    model2 = Sequential()
    model2.add(LSTM(lmst_nb, input_shape=(X_str_train.shape[1], X_str_train.shape[2])))
    model2.add(Dense(1))
    model2.compile(optimizer='adam', loss='mean_squared_error')
    # epochs=20 表示模型将遍历整个训练集 20 次。 batch_size=31 表示每个训练批次的样本数为 31
    model2.fit(X_str_train, y_train, epochs=20, batch_size=31, validation_data=(X_str_test,y_test))
    # model2.fit(X_str_train, Y_str_train, epochs=20, batch_size=32, validation_split=(X_str_train, Y_str_train))

    # Model evaluation
    loss2 = model2.evaluate(X_str_test, y_test)
    print(f'Loss (Mean Squared Error) sur l\'ensemble de test : {loss2}')

    # Prediction on training and test sets
    Y_train_predict = model2.predict(X_str_train)
    Y_test_predict = model2.predict(X_str_test)

    # calculate root mean squared error
    train_rmse = math.sqrt(mean_squared_error(y_train, Y_train_predict[:,0]))
    # RMSE为均方根误差
    print('Train Score: %.2f RMSE' % (train_rmse))
    test_rmse = math.sqrt(mean_squared_error(y_test, Y_test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % (test_rmse))

    # Calculate R2，决定系数
    # r2_score为判定系数介于负无穷-1之间，越大越好
    train_r2 = r2_score(y_train, Y_train_predict)
    print('Train Score: %.2f R2' % (train_r2))
    test_r2 = r2_score(y_test, Y_test_predict)
    print('Test Score: %.2f R2' % (test_r2))

    # Calculate the RMSE and R² std
    rmse_std = np.std([train_rmse, test_rmse])
    r2_std = np.std([train_r2, test_r2])
    print(f'Standard Deviation of RMSE: {rmse_std:.2f}')
    print(f'Standard Deviation of R2: {r2_std:.2f}')

    # Find non-standardized daily_sales
    # 将数据反标准化，进行绘图
    Y_str_train = (y_train * daily_sales_std) + daily_sales_mean
    Y_str_test = (y_test * daily_sales_std) + daily_sales_mean
    Y_train_predict[:, 0] = (Y_train_predict[:, 0] * daily_sales_std) + daily_sales_mean
    Y_test_predict[:, 0] = (Y_test_predict[:, 0] * daily_sales_std) + daily_sales_mean

    # Plot for training set
    plt.figure(figsize=(20, 6))
    plt.subplot(2, 1, 1)
    plt.plot( date_train,Y_str_train, label='Real values')
    plt.plot( date_train,Y_train_predict, label='Predicted values')
    plt.title(f'Comparison between Simple LSTM predicted/real values on the train',fontsize=14)
    plt.legend(fontsize=14)

    # Plot for testing set
    plt.subplot(2, 1, 2)
    plt.plot(date_test,Y_str_test, label='Real values')
    plt.plot(date_test,Y_test_predict, label='Predicted values')
    plt.title(f'Comparison between Simple LSTM predicted/real values on the test',fontsize=14)
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
def make_store_prediction_RNN(lmst_nb):
    X_str_train = x_train.to_numpy()
    X_str_train = X_str_train.reshape((X_str_train.shape[0], 1, X_str_train.shape[1]))
    X_str_test =x_test.to_numpy()
    X_str_test = X_str_test.reshape((X_str_test.shape[0], 1, X_str_test.shape[1]))
    model2 = Sequential()
    model2.add(SimpleRNN(lmst_nb, input_shape=(X_str_train.shape[1], X_str_train.shape[2])))
    model2.add(Dense(1))
    model2.compile(optimizer='adam', loss='mean_squared_error')
    model2.fit(X_str_train, y_train, epochs=20, batch_size=31, validation_data=(X_str_test, y_test))

    loss2 = model2.evaluate(X_str_test, y_test)
    print(f'Loss (Mean Squared Error) sur l\'ensemble de test : {loss2}')

    # Prediction on training and test sets
    Y_train_predict = model2.predict(X_str_train)
    Y_test_predict = model2.predict(X_str_test)

    # calculate root mean squared error
    train_rmse = math.sqrt(mean_squared_error(y_train, Y_train_predict[:, 0]))
    print('Train Score: %.2f RMSE' % (train_rmse))
    test_rmse = math.sqrt(mean_squared_error(y_test, Y_test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % (test_rmse))

    # Calculate R2
    train_r2 = r2_score(y_train, Y_train_predict)
    print('Train Score: %.2f R2' % (train_r2))
    test_r2 = r2_score(y_test, Y_test_predict)
    print('Test Score: %.2f R2' % (test_r2))

    # Calculate the RMSE and R² std
    rmse_std = np.std([train_rmse, test_rmse])
    r2_std = np.std([train_r2, test_r2])
    print(f'Standard Deviation of RMSE: {rmse_std:.2f}')
    print(f'Standard Deviation of R2: {r2_std:.2f}')

    # Find non-standardized daily_sales
    Y_str_train = (y_train * daily_sales_std) + daily_sales_mean
    Y_str_test = (y_test * daily_sales_std) + daily_sales_mean
    Y_train_predict[:, 0] = (Y_train_predict[:, 0] * daily_sales_std) + daily_sales_mean
    Y_test_predict[:, 0] = (Y_test_predict[:, 0] * daily_sales_std) + daily_sales_mean

    # Plot for training set
    plt.figure(figsize=(20, 6))
    plt.subplot(2, 1, 1)
    plt.plot(date_train,Y_str_train, label='Real values')
    plt.plot(date_train,Y_train_predict, label='Predicted values')
    plt.title(f'Comparison between Simple RNN predicted/real values on the train',fontsize=14)
    plt.legend(fontsize=14)

    # Plot for testing set
    plt.subplot(2, 1, 2)
    plt.plot(date_test,Y_str_test, label='Real values')
    plt.plot(date_test,Y_test_predict, label='Predicted values')
    plt.title(f'Comparison between Simple RNN predicted/real values on the test',fontsize=14)
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
def make_store_prediction_GRU(lmst_nb):
    # Reshape the variables for the GRU model
    X_str_train = x_train.to_numpy()
    X_str_train = X_str_train.reshape((X_str_train.shape[0], 1, X_str_train.shape[1]))
    X_str_test = x_test.to_numpy()
    X_str_test = X_str_test.reshape((X_str_test.shape[0], 1, X_str_test.shape[1]))

    # Construction of the GRU NN
    model2 = Sequential()
    model2.add(GRU(lmst_nb, input_shape=(X_str_train.shape[1], X_str_train.shape[2])))
    model2.add(Dense(1))
    model2.compile(optimizer='adam', loss='mean_squared_error')
    model2.fit(X_str_train, y_train, epochs=20, batch_size=31, validation_data=(X_str_test, y_test))

    # Model evaluation
    loss2 = model2.evaluate(X_str_test, y_test)
    print(f'Loss (Mean Squared Error) sur l\'ensemble de test : {loss2}')

    # Prediction on training and test sets
    Y_train_predict = model2.predict(X_str_train)
    Y_test_predict = model2.predict(X_str_test)

    # calculate root mean squared error
    train_rmse = math.sqrt(mean_squared_error(y_train, Y_train_predict[:, 0]))
    print('Train Score: %.2f RMSE' % (train_rmse))
    test_rmse = math.sqrt(mean_squared_error(y_test, Y_test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % (test_rmse))

    # Calculate R2
    train_r2 = r2_score(y_train, Y_train_predict)
    print('Train Score: %.2f R2' % (train_r2))
    test_r2 = r2_score(y_test, Y_test_predict)
    print('Test Score: %.2f R2' % (test_r2))

    # Calculate the RMSE and R² std
    rmse_std = np.std([train_rmse, test_rmse])
    r2_std = np.std([train_r2, test_r2])
    print(f'Standard Deviation of RMSE: {rmse_std:.2f}')
    print(f'Standard Deviation of R2: {r2_std:.2f}')

    # Find non-standardized daily_sales
    Y_str_train = (y_train * daily_sales_std) + daily_sales_mean
    Y_str_test = (y_test * daily_sales_std) + daily_sales_mean
    Y_train_predict[:, 0] = (Y_train_predict[:, 0] * daily_sales_std) + daily_sales_mean
    Y_test_predict[:, 0] = (Y_test_predict[:, 0] * daily_sales_std) + daily_sales_mean

    # Plot for training set
    plt.figure(figsize=(20, 6))
    plt.subplot(2, 1, 1)
    plt.plot(date_train,Y_str_train, label='Real values')
    plt.plot(date_train,Y_train_predict, label='Predicted values')
    plt.title(f'Comparison between Simple GRU predicted/real values on the train',fontsize=14)
    plt.legend(fontsize=14)
    print(Y_test_predict.shape[0])
    # Plot for testing set
    plt.subplot(2, 1, 2)
    plt.plot(date_test,Y_str_test, label='Real values')
    plt.plot(date_test,Y_test_predict, label='Predicted values')
    plt.title(f'Comparison between Simple GRU predicted/real values on the test',fontsize=14)
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

make_store_prediction_LSTM(256)
# Make daily sales predict16ions for the store of your choice and the number of Simple RNN neurons of your choice

make_store_prediction_RNN(256)
# %%
# Make daily sales predictions for the store of your choice and the number of GRU neurons of your choice

make_store_prediction_GRU(256)