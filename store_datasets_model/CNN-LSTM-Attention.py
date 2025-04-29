import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#设置随机种子保证每次结果一致
#torch.manual_seed(42)
#np.random.seed(42)
#random.seed(42)
pd.set_option('display.max_columns',None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')
dataset=pd.read_csv("../datasets/final_dataset.csv")
dataset=dataset.drop((dataset[(dataset['month'] == 1) & (dataset['day_of_month'] == 1)].index))
#预测出的销售额反归一化时使用
daily_sales_mean=dataset['daily_sales'].mean()
daily_sales_std=dataset['daily_sales'].std()
#将选中的列进行标准化
select_dataset=dataset[['daily_sales','daily_transactions','daily_onpromotion','dcoilwtico_interpolated']]
normalized_dataset=(select_dataset-select_dataset.mean())/select_dataset.std()
dataset[['daily_sales','daily_transactions','daily_onpromotion','dcoilwtico_interpolated']]=normalized_dataset
#获取指定门店的数据 数据共有1672行
dataset=dataset[dataset['store_nbr'] == 1]
train=dataset.iloc[:1310]
test=dataset.iloc[1310:]
#保存时间用于画图
datetime_for_plt_train=pd.to_datetime(train['date'])
datetime_for_plt_test=pd.to_datetime(test['date'])
#删除date和store_nbr
train=train.drop('date',axis=1)
test=test.drop('date',axis=1)
train=train.drop('store_nbr',axis=1)
test=test.drop('store_nbr',axis=1)
#构建训练集和测试集
x_train=train.drop('daily_sales',axis=1)
y_train=train['daily_sales']
x_test=test.drop('daily_sales',axis=1)
y_test=test['daily_sales']
x_train=x_train.values
y_train=y_train.values
x_test=x_test.values
y_test=y_test.values
x_train=x_train.reshape(1310,10,1)
y_train=y_train.reshape(1310,1)
x_test=x_test.reshape(362,10,1)
y_test=y_test.reshape(362,1)

x_train_tensor=torch.from_numpy(x_train)
x_train_tensor = x_train_tensor.to(torch.float32)
y_train_tensor=torch.from_numpy(y_train)
y_train_tensor = y_train_tensor.to(torch.float32)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # Assume x has shape (batch_size, sequence_length, hidden_size)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(weights, v)

        return attended_values

class CNNLSTM(nn.Module):
    def __init__(self, input_size,output_size, hidden_size, num_layers):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=1)
        self.batch1 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=1)
        self.batch2 = nn.BatchNorm1d(32)
        self.LSTM = nn.LSTM(input_size=5, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(32 * hidden_size, output_size)
        # self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        # in_size1 = x.size(0)  # one batch
        x = F.selu(self.conv1(x))
        x = self.conv2(x)
        x = F.selu(self.batch1(x))
        x = self.conv3(x)
        x = F.selu(self.batch2(x))
        x, h = self.LSTM(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        # in_size1 = x.size(0)  # one batch
        # x = x.view(in_size1, -1)
        # flatten the tensor x[:, -1, :]
        x = self.fc1(x)
        #output = torch.sigmoid(x)
        # output = self.fc2(x)
        return x


input_size=10
output_size=1
hidden_size=64
num_layers=3

model = CNNLSTM(input_size, output_size,hidden_size, num_layers)
print(model)

num_epochs = 1000
learning_rate = 0.001
criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
loss_list = []
# Train the model
for epoch in range(num_epochs):
    outputs = model(x_train_tensor)
    optimizer.zero_grad()

    # obtain the loss function
    loss = criterion(outputs, y_train_tensor)
    loss_list.append(loss)
    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


x_test_tensor=torch.from_numpy(x_test)
x_test_tensor = x_test_tensor.to(torch.float32)
Y_test_predict = model(x_test_tensor)
Y_train_predict = model(x_train_tensor)
print(Y_train_predict,Y_test_predict)
# calculate root mean squared error
train_rmse = math.sqrt(mean_squared_error(y_train, Y_train_predict.detach().numpy()[:,0]))
#RMSE为均方根误差
print('Train Score: %.2f RMSE' % (train_rmse))
test_rmse = math.sqrt(mean_squared_error(y_test, Y_test_predict.detach().numpy()[:,0]))
print('Test Score: %.2f RMSE' % (test_rmse))

# Calculate R2，决定系数
#r2_score为判定系数介于负无穷-1之间，越大越好
train_r2 = r2_score(y_train, Y_train_predict.detach().numpy())
print('Train Score: %.2f R2' % (train_r2))
test_r2 = r2_score(y_test, Y_test_predict.detach().numpy())
print('Test Score: %.2f R2' % (test_r2))

# Calculate the RMSE and R² std
rmse_std = np.std([train_rmse, test_rmse])
r2_std = np.std([train_r2, test_r2])
print(f'Standard Deviation of RMSE: {rmse_std:.2f}')
print(f'Standard Deviation of R2: {r2_std:.2f}')

# Find non-standardized daily_sales
#将数据反标准化，进行绘图
Y_str_train = (y_train * daily_sales_std) + daily_sales_mean
Y_str_test = (y_test * daily_sales_std) + daily_sales_mean
Y_train_predict[:,0] = (Y_train_predict[:,0] * daily_sales_std) + daily_sales_mean
Y_test_predict[:,0] = (Y_test_predict[:,0] * daily_sales_std) + daily_sales_mean

#  Plot for training set
plt.figure(figsize=(20, 6))
plt.subplot(2, 1, 1)
Y_str_train=Y_str_train.squeeze()
Y_train_predict=Y_train_predict.squeeze()
Y_str_test=Y_str_test.squeeze()
Y_test_predict=Y_test_predict.squeeze()

print(type(datetime_for_plt_train),datetime_for_plt_train.shape)
print(type(Y_str_train),Y_str_train.shape)
plt.plot(datetime_for_plt_train, Y_str_train, label='Real values (y_train)')
plt.plot(datetime_for_plt_train, Y_train_predict.detach().numpy(), label='Predicted values (Y_train_predict)')
plt.title(f'Comparison between CNN-LSTM-Attention predicted/real values on the train for the store 1')
plt.legend()

# Plot for testing set
plt.subplot(2, 1, 2)
plt.plot(datetime_for_plt_test, Y_str_test, label='Real values (y_test)')
plt.plot(datetime_for_plt_test, Y_test_predict.detach().numpy(), label='Predicted values (Y_test_predict)')
plt.title(f'Comparison between CNN-LSTM-Attention predicted/real values on the test for the store 1')
plt.legend()

plt.tight_layout()
plt.show()
