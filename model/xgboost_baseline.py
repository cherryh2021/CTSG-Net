import pandas as pd
import xgboost as xgb
import numpy as np
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('/home/hexingyi/GNN/data_nonpeak/combined_file.csv', index_col=0)
#print("df", df)
time_steps = 20  # 用于预测的时间步数
horizon = 6  # 往后预测的时间步数

def create_dataset(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset) - time_steps - horizon):
        a = dataset[i:(i + time_steps), :]
        X.append(a.reshape(-1))
        Y.append(dataset[i + time_steps:i + time_steps + horizon, :])
    return np.array(X), np.array(Y)


# 使用原始数据
data = df.values
# 数据拆分
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.1)
test_size = len(data) - train_size - val_size

X_all, Y_all = create_dataset(data, time_steps)
#print("X_all.shape, Y_all.shape", X_all.shape, Y_all.shape)
X_train, X_val, X_test = np.split(X_all, [train_size, train_size + val_size])
Y_train, Y_val, Y_test = np.split(Y_all, [train_size, train_size + val_size])  # 修改为 Y_all
#print("X_test.shape, Y_test.shape", X_test.shape, Y_test.shape)
Y_train = Y_train.reshape(Y_train.shape[0], -1)  
Y_val = Y_val.reshape(Y_val.shape[0], -1) 
Y_test = Y_test.reshape(Y_test.shape[0], -1)
print("Y_test.shape", Y_test.shape)

# 在构造函数中设置 eval_metric
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1, verbosity=1, eval_metric="mae")

# 提供验证集并训练模型
eval_set = [(X_train, Y_train), (X_val, Y_val)]
model.fit(X_train, Y_train, eval_set=eval_set, verbose=True)

# 预测
prediction = model.predict(X_test)
prediction = prediction.reshape(prediction.shape[0], horizon, -1)
Y_test = Y_test.reshape(Y_test.shape[0], horizon, -1)
print(prediction.shape)
print_out_allmae = []
print_out_nonmae = []
print_out_allrmse = []
print_out_nonrmse = []
print_out_largemae = []
print_out_largermse = []
print_out = [0, 1, 2]
nonzero_mae_hor = []
# 对每个时间步计算 MAE 和 RMSE
for i in range(horizon):
    non_zerotest = []
    non_zeropred = []
    large_test = []
    large_pred = []
    all_test = []
    all_pred = []
    for k in range(len(Y_test) - horizon):
        for j in range(prediction.shape[2]):
            #print(Y_test[k + i][j])
            all_test.append(Y_test[k][i][j])
            all_pred.append(prediction[k][i][j])
            if Y_test[k][i][j] != 0:
                non_zerotest.append(Y_test[k][i][j])
                non_zeropred.append(prediction[k][i][j])
            if Y_test[k][i][j] > 100:
                large_test.append(Y_test[k][i][j])
                large_pred.append(prediction[k][i][j])
    mae = mean_absolute_error(all_test, all_pred)
    rmse = np.sqrt(mean_squared_error(all_test, all_pred))
    non_zeromae = mean_absolute_error(non_zerotest, non_zeropred)
    non_zerormse = np.sqrt(mean_squared_error(non_zerotest, non_zeropred))
    largemae = mean_absolute_error(large_test, large_pred)
    largermse = np.sqrt(mean_squared_error(large_test, large_pred))
    nonzero_mae_hor.append(non_zeromae)
    print('Horizon: {} | MAE(all data): {:.4f} | RMSE(all data): {:.4f} | MAE(non-zero data): {:.4f} | RMSE(non-zero data): {:.4f} | MAE(>100m data): {:.4f} | RMSE(>100m data): {:.4f}'.format(i+1, mae, rmse, non_zeromae, non_zerormse, largemae, largermse))
    if i in print_out:
        print_out_allmae.append(mae)
        print_out_nonmae.append(non_zeromae)
        print_out_allrmse.append(rmse)
        print_out_nonrmse.append(non_zerormse)
        print_out_largemae.append(largemae)
        print_out_largermse.append(largermse)
		
print("all data")
log = '5sMAE | 10sMAE | 15sMAE | 5sRMSE | 10sRMSE | 15sRMSE'
print(log)
print("{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(print_out_allmae[0], print_out_allmae[1], print_out_allmae[2], print_out_allrmse[0], print_out_allrmse[1], print_out_allrmse[2]))
print("non-zero data")
log = '5sMAE | 10sMAE | 15sMAE | 5sRMSE | 10sRMSE | 15sRMSE'
print(log)
print("{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(print_out_nonmae[0], print_out_nonmae[1], print_out_nonmae[2], print_out_nonrmse[0], print_out_nonrmse[1], print_out_nonrmse[2]))
print(">100m data")
log = '5sMAE | 10sMAE | 15sMAE | 5sRMSE | 10sRMSE | 15sRMSE'
print(log)
print("{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(print_out_largemae[0], print_out_largemae[1], print_out_largemae[2], print_out_largermse[0], print_out_largermse[1], print_out_largermse[2]))
print("Training finished")
filename = '/home/hexingyi/GNN/output/' + 'XGBoost' + '.csv'
header = ['5sMAE(all data)','10sMAE(all data)','15sMAE(all data)','5sRMSE(all data)','10sRMSE(all data)','15sRMSE(all data)',
          '5sMAE(non-zero data)','10sMAE(non-zero data)','15sMAE(non-zero data)','5sRMSE(non-zero data)','10sRMSE(non-zero data)','15sRMSE(non-zero data)',
          '5sMAE(>100m data)','10sMAE(>100m data)','15sMAE(>100m data)','5sRMSE(>100m data)','10sRMSE(>100m data)','15sRMSE(>100m data)']
data = [[print_out_allmae[0], print_out_allmae[1], print_out_allmae[2], print_out_allrmse[0], print_out_allrmse[1], print_out_allrmse[2],
         print_out_nonmae[0], print_out_nonmae[1], print_out_nonmae[2], print_out_nonrmse[0], print_out_nonrmse[1], print_out_nonrmse[2],
         print_out_largemae[0], print_out_largemae[1], print_out_largemae[2], print_out_largermse[0], print_out_largermse[1], print_out_largermse[2]]]
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for row in data:
        writer.writerow(row)
filename2 = '/home/hexingyi/GNN/output_horizon/' + 'XGBoost' + '.csv'
header = ['5s', '10s', '15s', '20s', '25s', '30s']
data = [[nonzero_mae_hor[0], nonzero_mae_hor[1], nonzero_mae_hor[2], nonzero_mae_hor[3],
         nonzero_mae_hor[4], nonzero_mae_hor[5]]]
with open(filename2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for row in data:
        writer.writerow(row)
