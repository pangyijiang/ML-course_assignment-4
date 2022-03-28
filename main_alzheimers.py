import scipy.io
from LogisticRegressionL1 import LR_L1
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


file_path_dataset = "./alzheimers/ad_data.mat"
file_path_n = "./alzheimers/feature_name.mat"

dataset = scipy.io.loadmat(file_path_dataset)
data_name = scipy.io.loadmat(file_path_n)

x_train, y_train, x_test, y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]
y_train[y_train < 0] = 0
y_test[y_test < 0] = 0

acc_trace = []
for reg_rate in [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    LR = LR_L1(dim_in = x_train.shape[1], max_iter = 500, reg_rate = reg_rate, batch_size = 32, lr = 1e-3)
    loss_trace = LR.fit(x_train, y_train)
    Y_pred = LR.predict(x_test)
    r_avg_acc = [1 if i ==j else 0 for i, j in zip(Y_pred, y_test)]
    avg_acc = np.sum(r_avg_acc)/len(r_avg_acc)
    # print("acc = ", avg_acc)
    acc_trace.append(avg_acc)
    fpr, tpr, threshold = metrics.roc_curve(y_test, Y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print("acc. = %.3f, AUC = %.4f, num of param(>0.005) = %d" % (avg_acc, roc_auc, np.sum([1 for i in LR.theta_Set if i > 0.005])))

x_plot, y_plot =  range(len(acc_trace)), acc_trace
plt.plot(x_plot, y_plot, marker = "+")
for i,j in zip(x_plot,y_plot):
    plt.annotate("%.4f"%j,xy=(i,j))
plt.xlabel("regularization strength")
plt.ylabel("Acc.")
# plt.ylim([0.8, 1])
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], ["0.0", "00.1", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9","1.0"])
plt.show()