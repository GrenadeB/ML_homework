from utils import *
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut

datas, labels = read_data('./AALCorrArray_BJ_76_Class.mat', './Age_BJ_76_Class.mat')
pca = PCA()
low_datas = pca.fit_transform(datas, 200)
print(low_datas.shape)

acc = 0
loop = LeaveOneOut()
for train_index, test_index in loop.split(low_datas, labels):
    train_datas = low_datas[train_index]
    train_labels = labels[train_index]
    test_data = low_datas[test_index]
    test_label = labels[test_index]
    model = SVC()
    model.fit(train_datas, train_labels)
    predict_label = model.predict(test_data)
    print('true label:', test_label)
    print('predict label:', predict_label)
    acc = acc + np.equal(predict_label, test_label)
acc = acc/76
print('Accuracy:', acc)