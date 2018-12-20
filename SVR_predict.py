from utils import *


datas_mat, labels_mat = read_data('./AALCorrArray_BJ_63.mat', './Age_BJ_63.mat')
datas = datas_mat['Data']
labels = labels_mat['Label']

pac = PCA()
low_datas = pac.fit_transform(datas, 200)

print(low_datas.shape)