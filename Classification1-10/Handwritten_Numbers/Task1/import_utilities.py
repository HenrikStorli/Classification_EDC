import numpy as np
from scipy import io as io

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

def import_data_subset(path, n):
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
        data = np.zeros((n, row_size*col_size))

        for i in range(n):        #Kun for test
            for j in range(row_size*col_size):
                data[i,j] = np.fromfile(fid, dtype=np.uint8, count=1)
        return data

def import_labels_subset(path,n):
    """
    Import labels from .bin files
    returns ndarray
    """

    with open(path, 'r') as fid:
        dt = np.dtype('>i4')
        magic_num = np.fromfile(fid, dtype=dt, count=1)[-1]
        num = np.fromfile(fid, dtype=dt, count=1)[-1]
        data = np.zeros((n,1))
        for i in range(n):        #
            data[i] = np.fromfile(fid, dtype=np.uint8, count=1)
        return data

def import_all_subset(n):
    """Returns datasets and labels"""

    test_images = import_data_subset('.\\data\\test_images.bin',n)
    test_labels = import_labels_subset('.\\data\\test_labels.bin',n)
    train_images = import_data_subset('.\\data\\train_images.bin',n)
    train_labels = import_labels_subset('.\\data\\train_labels.bin',n)

    return test_images, test_labels, train_images, train_labels


def import_data_subset_to_file(path, n):
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
        data = np.zeros((n, row_size*col_size))

        for i in range(n):        #Kun for test
            for j in range(row_size*col_size):
                data[i,j] = np.fromfile(fid, dtype=np.uint8, count=1)

        return data

def import_labels_subset_to_file(path,n):
    """
    Import labels from .bin files
    returns ndarray
    """

    with open(path, 'r') as fid:
        dt = np.dtype('>i4')
        magic_num = np.fromfile(fid, dtype=dt, count=1)[-1]
        num = np.fromfile(fid, dtype=dt, count=1)[-1]
        data = np.zeros((n,1))
        for i in range(n):        #
            data[i] = np.fromfile(fid, dtype=np.uint8, count=1)

        return data



def import_all_subset_to_file(n):
    """Returns datasets and labels"""

    test_images = import_data_subset('.\\data\\test_images.bin',n)
    test_labels = import_labels_subset('.\\data\\test_labels.bin',n)
    train_images = import_data_subset('.\\data\\train_images.bin',n)
    train_labels = import_labels_subset('.\\data\\train_labels.bin',n)

    data_dict = {
        'test_images': test_images,
        'test_labels': test_labels,
        'train_images': train_images,
        'train_labels': train_labels,
    }
    io.savemat('.\\mat_files\\MNIST.mat', data_dict)
