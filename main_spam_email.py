import numpy as np
from LogisticRegression import LR_customized
import matplotlib.pyplot as plt


file_path_x = "./spam_email/data.txt"
file_path_y = "./spam_email/labels.txt"

def load_data(file_path):
    data_x = []
    with open(file_path, mode = 'r') as f:
        lines = f.readlines()
        for l in lines:
            X = l.split(" ")
            X = [float(x) for x in X if x != ""]
            data_x.append(X)
    data_x = np.array(data_x)
    return data_x

data_x = load_data(file_path_x)
data_y = load_data(file_path_y)

data_x_train, data_y_train = data_x[:2001], data_y[:2001]
data_x_test, data_y_test = data_x[2001:], data_y[2001:]


acc_trace = []
for n_iter in [200, 500, 800, 1000, 1500, 2000]:
    LR = LR_customized(dim_in = data_x_train.shape[1], max_iter = n_iter)
    loss_trace = LR.fit(data_x_train, data_y_train)
    Y_pred = LR.predict(data_x_test)
    r_avg_acc = [1 if i ==j else 0 for i, j in zip(Y_pred, data_y_test)]
    avg_acc = np.sum(r_avg_acc)/len(r_avg_acc)
    print("acc = ", avg_acc)
    acc_trace.append(avg_acc)

x_plot, y_plot =  range(len(acc_trace)), acc_trace
plt.plot(x_plot, y_plot, marker = "+")
for i,j in zip(x_plot,y_plot):
    plt.annotate("%.4f"%j,xy=(i,j))
plt.xlabel("Number of iteration")
plt.ylabel("Acc.")
plt.ylim([0.8, 1])
plt.xticks([0,1,2,3,4,5], ["n_iter = 200", "500", "800", "1000", "1500", "2000"])
plt.show()