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