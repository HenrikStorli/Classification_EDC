#from sklearn.cluster import KMeans
import numpy as np

#from scipy.cluster.vq import vq, kmeans, whiten

def sort_training_images_into_classes(training_images, training_labels):
    num_rows_training, num_cols_training = training_images.shape

    sorted_images = np.zeros((10, 6000, 784))

    for k in range(num_rows_training):
        class_number = training_labels[k,0]
