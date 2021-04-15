import numpy as np

def NN_classifier(test_set, training_set, class_vector_training_set):

    num_rows_training, num_cols_training = training_set.shape
    num_rows_test, num_cols_test = test_set.shape

    distance_matrix = np.zeros((num_rows_test,num_rows_training))

    for i in range(num_rows_test):
        for j in range(num_rows_training):
            distance_matrix[i,j] = euclidian_distance(test_set[i,:], training_set[j,:])

    class_vector_guess = np.zeros((num_rows_test,1))


    for k in range(num_rows_test):
        min_index = np.argmin(np.abs(distance_matrix[k,:]))
        class_vector_guess[k] = class_vector_training_set[min_index]

    return class_vector_guess


def euclidian_distance(x,y):
    return np.multiply((x-y),np.transpose(x-y))

def import_data(path):
    """
    Import datasets from .bin files
    returns ndarray
    """

    with open(path, 'r') as fid:
        dt = np.dtype('>i4')
        magic_num = np.fromfile(fid, dtype=dt, count=1)[-1]
        num_test = np.fromfile(fid, dtype=dt, count=1)[-1]
        row_size = np.fromfile(fid, dtype=dt, count=1)[-1]
        col_size = np.fromfile(fid, dtype=dt, count=1)[-1]
        data = np.zeros((num_test, row_size*col_size))

        for i in range(int(num_test)):        #Kun for test
            for j in range(row_size*col_size):
                data[i,j] = np.fromfile(fid, dtype=np.uint8, count=1)
        return data

def import_labels(path):
    """
    Import labels from .bin files
    returns ndarray
    """

    with open(path, 'r') as fid:
        dt = np.dtype('>i4')
        magic_num = np.fromfile(fid, dtype=dt, count=1)[-1]
        num = np.fromfile(fid, dtype=dt, count=1)[-1]
        data = np.zeros((num,1))
        for i in range(int(num)):        #
            data[i] = np.fromfile(fid, dtype=np.uint8, count=1)
        return data

def import_all():
    """Returns datasets and labels"""

    test_images = import_data('.\\data\\test_images.bin')
    test_labels = import_labels('.\\data\\test_labels.bin')
    train_images = import_data('.\\data\\train_images.bin')
    train_labels = import_labels('.\\data\\train_labels.bin')

    return test_images, test_labels, train_images, train_labels

import_all()