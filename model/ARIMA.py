import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# 加载数据
df = pd.read_csv('/home/hexingyi/GNN/data_nonpeak/combined_file.csv', index_col=0)
data = df.values

time_steps = 20  # 用于预测的时间步数
horizon = 6  # 往后预测的时间步数


# 创建数据集函数
def create_dataset(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset) - time_steps - horizon):
        a = dataset[i:(i + time_steps), :]
        X.append(a)
        Y.append(dataset[i + time_steps:i + time_steps + horizon, :])
    return np.array(X), np.array(Y)


# 数据拆分
X_all, Y_all = create_dataset(data, time_steps)
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.1)
test_size = len(data) - train_size - val_size
X_train, X_val, X_test = np.split(X_all, [train_size, train_size + val_size])
Y_train, Y_val, Y_test = np.split(Y_all, [train_size, train_size + val_size])
print("X_test.shape", X_test.shape, "Y_test.shape", Y_test.shape)  # (8614, 20, 351), (8614, 6, 351)
# 初始化性能评估数组
all_mae = []
all_rmse = []
nonzero_mae = []
nonzero_rmse = []
large_mae = []
large_rmse = []
print_out_allmae = []
print_out_nonmae = []
print_out_allrmse = []
print_out_nonrmse = []
print_out_largemae = []
print_out_largermse = []
print_out = [0, 1, 2]
# 对每个节点训练ARIMA模型并预测
predictions = []
for node_index in range(data.shape[1]):
    print("node_index", node_index)
    t1 = time.time()
    node_predictions = []
    for x in X_test[:, :, node_index]:
        # print("x.shape", x.shape)  # (20, )
        model = ARIMA(x, order=(1, 2, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon)
        node_predictions.append(forecast)
    t2 = time.time()
    print("Time:{:.4f}".format(t2-t1))
    predictions.append(node_predictions)
    print("predictions.shape", np.array(predictions).shape)


predictions = np.array(predictions).transpose(1, 2, 0)  # 调整形状以便与Y_test匹配
print("predictions.shape", predictions.shape)
# 按时间步分析预测结果
for t in range(horizon):
    y_true = Y_test[:, t, :].flatten()
    y_pred = predictions[:, t, :].flatten()

    # 所有数据
    mae_all = mean_absolute_error(y_true, y_pred)
    rmse_all = np.sqrt(mean_squared_error(y_true, y_pred))
    all_mae.append(mae_all)
    all_rmse.append(rmse_all)

    # 非零数据
    nonzero_mask = y_true != 0
    mae_nonzero = mean_absolute_error(y_true[nonzero_mask], y_pred[nonzero_mask])
    rmse_nonzero = np.sqrt(mean_squared_error(y_true[nonzero_mask], y_pred[nonzero_mask]))
    nonzero_mae.append(mae_nonzero)
    nonzero_rmse.append(rmse_nonzero)

    # 大于100数据
    large_mask = y_true > 100
    mae_large = mean_absolute_error(y_true[large_mask], y_pred[large_mask])
    rmse_large = np.sqrt(mean_squared_error(y_true[large_mask], y_pred[large_mask]))
    large_mae.append(mae_large)
    large_rmse.append(rmse_large)

    print(f'Horizon {t+1}: MAE (all data) = {mae_all:.4f}, RMSE (all data) = {rmse_all:.4f}, MAE (non-zero data) = {mae_nonzero:.4f}, RMSE (non-zero data) = {rmse_nonzero:.4f}, MAE (>100 data) = {mae_large:.4f}, RMSE (>100 data) = {rmse_large:.4f}')

    if t in print_out:
        print_out_allmae.append(mae_all)
        print_out_nonmae.append(mae_nonzero)
        print_out_allrmse.append(rmse_all)
        print_out_nonrmse.append(rmse_nonzero)
        print_out_largemae.append(mae_large)
        print_out_largermse.append(rmse_large)

# 汇总性能评估信息
print("all data")
log = '5sMAE | 10sMAE | 15sMAE | 5sRMSE | 10sRMSE | 15sRMSE'
print(log)
print("{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".
      format(print_out_allmae[0], print_out_allmae[1], print_out_allmae[2],
             print_out_allrmse[0], print_out_allrmse[1], print_out_allrmse[2]))
print("non-zero data")
log = '5sMAE | 10sMAE | 15sMAE | 5sRMSE | 10sRMSE | 15sRMSE'
print(log)
print("{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".
      format(print_out_nonmae[0], print_out_nonmae[1], print_out_nonmae[2],
             print_out_nonrmse[0], print_out_nonrmse[1], print_out_nonrmse[2]))
print(">100m data")
log = '5sMAE | 10sMAE | 15sMAE | 5sRMSE | 10sRMSE | 15sRMSE'
print(log)
print("{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".
      format(print_out_largemae[0], print_out_largemae[1], print_out_largemae[2],
             print_out_largermse[0], print_out_largermse[1], print_out_largermse[2]))
