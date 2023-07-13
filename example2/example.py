import pandas as pd
import numpy as np
from sklearn.svm import SVC
from TCA import TCA

# source domain training data
Xs_y = pd.read_csv('Xs_y.csv')
Xs = np.array(Xs_y.iloc[:,:-1])
ys = np.array(Xs_y.iloc[:,-1])
# target domain training data
Xt_y = pd.read_csv('Xt_y.csv')
Xt = np.array(Xt_y.iloc[:,:-1])
yt = np.array(Xt_y.iloc[:,-1])
# test data
test = pd.read_csv('Xtest.csv')
Xtest = np.array(test.iloc[:,:-1])
ytest = np.array(test.iloc[:,-1])

reg = SVC()

def acc(y_pre, y):
    right = abs(y_pre - y).sum()/2
    return (len(y_pre) - right)/len(y_pre)
################################################################
# case one 
# training on target domina data (Xt_y.csv) and test on testing dataset (Xtest.csv)
pre_label = reg.fit(Xt, yt,).predict(Xtest)
rate = acc(pre_label,ytest)
print('Without transfer, we have acc rate %f' % rate )

# case two
# training on mapped space (4-d) with source (Xs_y.csv) and target domina data (Xt_y.csv), test on testing dataset (Xtest.csv)
acc_list = [rate]
numbers = np.linspace(0.01, 1, 10)
for j in range(len(numbers)):
    model = TCA(dim=1, lamda=numbers[j], gamma=0.5)
    Train_X = np.vstack((Xs,Xt))
    Xs_new, Xt_new = model.fit(Train_X,Xtest)
    Train_y = np.hstack((ys,yt))
    transfer_pre_label = reg.fit(Xs_new, Train_y).predict(Xt_new)
    transfer_rate = acc(transfer_pre_label,ytest)
    acc_list.append(transfer_rate)
    print('lamda = {}'.format(numbers[j]))
    print('With transfer, we have acc rate %f' % transfer_rate ,'\n')


# plot the result
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots()
x = numbers

plt.plot(x,  acc_list[1:],marker='o',linestyle='-', color='blue',label='TCA transfer with different lamda')
ax.axhline(y=acc_list[0], color='green', linestyle='--', label='without TCA transfer')
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel('lamda')
plt.ylabel('classification accuracy on test data')

ax.legend()
plt.savefig('iteration number.png',bbox_inches = 'tight',dpi=600)
plt.savefig('iteration number.svg',bbox_inches = 'tight',dpi=600)
plt.show()



