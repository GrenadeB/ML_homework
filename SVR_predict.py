from utils import *
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut

datas, labels = read_data('./AALCorrArray_BJ_63.mat', './Age_BJ_63.mat')

pac = PCA()
low_datas = pac.fit_transform(datas, 200)

print(low_datas.shape)
loop = LeaveOneOut()
for train_index, test_index in loop.split(low_datas, labels):
    train_datas = low_datas[train_index]
    train_labels = labels[train_index]
    test_data = low_datas[test_index]
    test_label = labels[test_index]
    model = SVR()
    model.fit(train_datas, train_labels)
    print(model.predict(test_data))
