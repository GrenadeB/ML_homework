import numpy as np
import scipy.io


class PCA(object):
    """
    the PCA class include two Attributes and three Methods
    """
    def __init__(self):
        self.x_mean = None
        self.main_vectors = None

    def fit_transform(self, train_datas, topK):
        """
        reduce dimension using PCA method with one parameter(topK)
        :param train_datas: train data used to calculate x_mean and main_vectors
        :param topK: a integer, dimension with reduction
        :return: 1.train data on reduced dimension with shape(-1, topK) 2.rebuilt train data with shape(-1, 2000)
        """
        self.x_mean = np.mean(train_datas, axis=0)
        datas = train_datas - self.x_mean
        cov_matrix = np.cov(datas, rowvar=0)
        eig_values, eig_vectors = np.linalg.eig(cov_matrix)
        # 对特征值从小到大排序
        sort_eig_index = np.argsort(eig_values)
        eig_index = sort_eig_index[:-(topK + 1):-1]
        self.main_vectors = eig_vectors[:, eig_index]
        x_low = datas.dot(self.main_vectors)
        return np.real(x_low)

    def transform(self, test_data):
        """
        reduce dimension using entity's attributes x_mean and main_vectors
        :param test_data: test_data with high dimension
        :return: 1.test data on reduced dimension 2.rebuilt test data
        """
        datas = test_data-self.x_mean
        x_low = datas.dot(self.main_vectors)
        return np.real(x_low)


def read_data(input_path, label_path):
    datas_mat = scipy.io.loadmat(input_path)
    labels_mat = scipy.io.loadmat(label_path)
    return datas_mat['Data'], labels_mat['Label']
