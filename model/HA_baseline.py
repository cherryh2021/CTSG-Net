import csv
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


large = 100
signal = []
with open('/home/hexingyi/GNN/data_nonpeak/siganl_nonpeak.csv', mode='r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        signal.extend(row)
signal_lengths = [int(x) for x in signal]
df = pd.read_csv('/home/hexingyi/GNN/data_nonpeak/combined_file.csv', index_col=0)
data = df.values
sample_num = len(data)
print("sample_num", sample_num)
test_num = 1708
train_num = sample_num - test_num
pred_hor = 6
seq_out = 6
avg_horizon = 20

horizon = list()
MAE_hor = []
RMSE_hor = []
nonzero_MAE_hor = []
nonzero_RMSE_hor = []
large_MAE_hor = []
large_RMSE_hor = []
print_out_allmae = []
print_out_nonmae = []
print_out_allrmse = []
print_out_nonrmse = []
print_out_largemae = []
print_out_largermse = []
print_out = [0, 1, 2]
test_true = []
test_pred = []
for horiz in range(seq_out):
    MAE_lane = []
    RMSE_lane = []
    nonzero_MAE_lane = []
    non_zero_RMSE_lane = []
    large_MAE_lane = []
    large_RMSE_lane = []
    for k in range(test_num - seq_out):
        for i in range(df.shape[1]):
            true = df.iloc[k + horiz + train_num, i]
            cycle = int(signal_lengths[i] / 5)
            pred = sum([df.iloc[k + horiz + train_num - j * cycle, i] for j in range(1, 1 + avg_horizon)]) / avg_horizon # 用历史20个周期的数据来做历史平均
            if i == 312 and horiz == 0:
                test_true.append(true)
                test_pred.append(pred)
            MAE_lane.append(abs(pred - true))
            RMSE_lane.append((pred - true) ** 2)
            if true != 0:
                nonzero_MAE_lane.append(abs(pred - true))
                non_zero_RMSE_lane.append((pred - true) ** 2)
            if true > large:
                large_MAE_lane.append(abs(pred - true))
                large_RMSE_lane.append((pred - true) ** 2)
    MAE_hor.append(np.mean(MAE_lane))
    RMSE_hor.append(math.sqrt(np.mean(RMSE_lane)))
    nonzero_MAE_hor.append(np.mean(nonzero_MAE_lane))
    nonzero_RMSE_hor.append(math.sqrt(np.mean(non_zero_RMSE_lane)))
    large_MAE_hor.append(np.mean(large_MAE_lane))
    large_RMSE_hor.append(math.sqrt(np.mean(large_RMSE_lane)))
for horiz in range(pred_hor):
    print('Horizon: {} | MAE(all data): {:.4f} | RMSE(all data): {:.4f} | MAE(non-zero data): {:.4f} | RMSE(non-zero data): {:.4f} | MAE(>100m data): {:.4f} | RMSE(>100m data): {:.4f}'.format(horiz + 1, MAE_hor[horiz], RMSE_hor[horiz], nonzero_MAE_hor[horiz], nonzero_RMSE_hor[horiz], large_MAE_hor[horiz], large_RMSE_hor[horiz]))
    if horiz in print_out:
        print_out_allmae.append(MAE_hor[horiz])
        print_out_nonmae.append(nonzero_MAE_hor[horiz])
        print_out_allrmse.append(RMSE_hor[horiz])
        print_out_nonrmse.append(nonzero_RMSE_hor[horiz])
        print_out_largemae.append(large_MAE_hor[horiz])
        print_out_largermse.append(large_RMSE_hor[horiz])
print("all data")
log = '5sMAE | 10sMAE | 15sMAE | 5sRMSE | 10sRMSE | 15sRMSE'
print(log)
print("{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(print_out_allmae[0], print_out_allmae[1], print_out_allmae[2], print_out_allrmse[0], print_out_allrmse[1], print_out_allrmse[2]))
print("non-zero data")
log = '5sMAE | 10sMAE | 15sMAE | 5sRMSE | 10sRMSE | 15sRMSE'
print(log)
print("{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(print_out_nonmae[0], print_out_nonmae[1], print_out_nonmae[2], print_out_nonrmse[0], print_out_nonrmse[1], print_out_nonrmse[2]))
print("Training finished")
print(">{}m data".format(large))
log = '5sMAE | 10sMAE | 15sMAE | 5sRMSE | 10sRMSE | 15sRMSE'
print(log)
print("{:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(print_out_largemae[0], print_out_largemae[1], print_out_largemae[2], print_out_largermse[0], print_out_largermse[1], print_out_largermse[2]))
print("Training finished")
filename = '/home/hexingyi/GNN/output/' + 'HA' + '.csv'
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
filename2 = '/home/hexingyi/GNN/output_horizon/' + 'HA' + '.csv'
header = ['5s', '10s', '15s', '20s', '25s', '30s']
data = [[nonzero_MAE_hor[0], nonzero_MAE_hor[1], nonzero_MAE_hor[2], nonzero_MAE_hor[3],
         nonzero_MAE_hor[4], nonzero_MAE_hor[5]]]
with open(filename2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for row in data:
        writer.writerow(row)
# visualize
mpl.rc('font', family='Times New Roman', size=16)  # 可以根据需要调整字体大小
x_index = list(range(len(test_pred)))
plt.figure(figsize=(30, 13))
plt.plot(x_index, test_true, label='True Value')
plt.plot(x_index, test_pred, label='Prediction', linestyle='--')
plt.figtext(0.1, 0.035, "Start: {} End: {}".format('2024-03-02 19:28:50', '2024-03-03 19:30:00'), ha="left",fontdict={'family': 'Times New Roman', 'fontsize': 20},
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=1'))
plt.legend(fontsize=30)
plt.title('Test Data', fontsize=32)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.grid(True)
# 更新文件保存路径，以保存为EPS格式
save_path = '/home/hexingyi/GNN/output_image/_'+ 'HA' + '_0_prediction_test.eps'
print("save picture to", save_path)

# 保存图像为EPS格式，这里我们使用默认的DPI，因为EPS是矢量格式，DPI设置不会影响最终质量
plt.savefig(save_path, format='eps')
