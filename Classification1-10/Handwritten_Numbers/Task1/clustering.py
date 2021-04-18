#from sklearn.cluster import KMeans
import numpy as np

#from scipy.cluster.vq import vq, kmeans, whiten

def sort_training_images_into_classes(training_images, training_labels):
    num_rows_training, num_cols_training = training_images.shape

    sorted_images = np.zeros((10, 100, 784))
    class_iter_count = np.zeros((10,1))

    for k in range(num_rows_training):
        class_number = int(training_labels[k,0])
        sorted_class_index = int(class_iter_count[class_number, 0])
        sorted_images[class_number, sorted_class_index, :] = training_images[k,:]

        class_iter_count[class_number, 0] += 1

    return sorted_images

