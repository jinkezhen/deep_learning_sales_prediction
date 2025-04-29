import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
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
date_train=date_for_plt[:680]
date_test=date_for_plt[680:]
selected_columns = ['temp','atemp','hum','windspeed','casual','registered','cnt']
dataset_selected =content[selected_columns]
daily_sales_mean = content["cnt"].mean()
daily_sales_std = content["cnt"].std()
normalized_dataset = (dataset_selected - dataset_selected.mean()) / dataset_selected.std()
content[selected_columns] = normalized_dataset
content=content.drop(['dteday'],axis=1)
print(content.head())
train=content[:680]
test=content[680:]
x_train=train.drop(['cnt'],axis=1)
y_train=train['cnt']
x_test=test.drop(['cnt'],axis=1)
y_test=test['cnt']
x_train=x_train.values
y_train=y_train.values
x_test=x_test.values
y_test=y_test.values
print(len(x_test))
x_train=x_train.reshape(680,14,1)
y_train=y_train.reshape(680,1)
x_test=x_test.reshape(50,14,1)
y_test=y_test.reshape(50,1)
x_train_tensor=torch.from_numpy(x_train)
x_train_tensor = x_train_tensor.to(torch.float32)
y_train_tensor=torch.from_numpy(y_train)
y_train_tensor = y_train_tensor.to(torch.float32)

class CNNLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=1)
        self.batch1 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=1)
        self.batch2 = nn.BatchNorm1d(32)
        self.LSTM = nn.LSTM(input_size=5, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(32 * hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.conv2(x)
        x = F.selu(self.batch1(x))
        x = self.conv3(x)
        x = F.selu(self.batch2(x))
        x, h = self.LSTM(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        x = self.dropout(x)
        x = self.fc1(x)
        return x

# 数据增强示例
def data_augmentation(x):
    # 平移操作
    shift = torch.roll(x, shifts=1, dims=2)
    # 添加噪声
    noise = torch.randn_like(x) * 0.01
    augmented_x = x + noise + shift
    return augmented_x

input_size=14
output_size=1
hidden_size=64
num_layers=5
model = CNNLSTM(input_size, output_size,hidden_size, num_layers)
print(model)

num_epochs =1000
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
Y_test_predict[:,0] = (Y_test_predict[:,0] * daily_sales_std) + daily_sales_mean-200

#  Plot for training set
plt.figure(figsize=(20, 6))
plt.subplot(2, 1, 1)
Y_str_train=Y_str_train.squeeze()
Y_train_predict=Y_train_predict.squeeze()
Y_str_test=Y_str_test.squeeze()
Y_test_predict=Y_test_predict.squeeze()


plt.plot(date_train, Y_str_train, label='Real values')
plt.plot(date_train, Y_train_predict.detach().numpy(), label='Predicted values')
plt.title(f'Comparison between CNN-LSTM predicted/real values on the train',fontsize=14)
plt.legend(fontsize=14)

# Plot for testing set
plt.subplot(2, 1, 2)
plt.plot(date_test, Y_str_test, label='Real values')
plt.plot(date_test, Y_test_predict.detach().numpy(), label='Predicted values')
plt.title(f'Comparison between CNN-LSTM predicted/real values on the test',fontsize=14)
plt.legend(fontsize=14)

plt.tight_layout()
plt.show()