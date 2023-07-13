import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
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

reg = DecisionTreeRegressor(max_depth=5,random_state=0)
################################################################
# case one 
# training on target domina data (Xt_y.csv) and test on testing dataset (Xtest.csv)
pre_label = reg.fit(Xt, yt,).predict(Xtest)
rate = np.sqrt(mean_squared_error(pre_label,ytest))
print('Without transfer, we have misclassify rate %f' % rate )

# case two
# training on mapped space (4-d) with source (Xs_y.csv) and target domina data (Xt_y.csv), test on testing dataset (Xtest.csv)
RMSE_list = [rate]
for j in range(20):
    model = TCA(dim=(20-j), lamda=.3, gamma=1)
    Train_X = np.vstack((Xs,Xt))
    Xs_new, Xt_new = model.fit(Train_X,Xtest)
    Train_y = np.hstack((ys,yt))
    transfer_pre_label = reg.fit(Xs_new, Train_y).predict(Xt_new)
    transfer_rate = np.sqrt(mean_squared_error(transfer_pre_label,ytest))
    RMSE_list.append(transfer_rate)
    print('{} dimension'.format((20-j)))
    print('With transfer, we have misclassify rate %f' % transfer_rate ,'\n')


# plot the result
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots()
x = range(len(RMSE_list)-1, 0, -1)

plt.plot(x, RMSE_list[1:], marker='o', linestyle='--', color='blue',label='with TCA transfer')
ax.axhline(y=RMSE_list[0], color='green', linestyle='--', label='without TCA transfer')
ax.set_xticks(x)
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel('TCA dimensions')
plt.ylabel('RMSE on test data')

ax.legend()
plt.savefig('iteration number.png',bbox_inches = 'tight',dpi=600)
plt.savefig('iteration number.svg',bbox_inches = 'tight',dpi=600)
plt.show()



