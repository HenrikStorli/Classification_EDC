import numpy as np


def NN_classifier(test_set, training_set, class_vector_training_set):
    num_rows_training, num_cols_training = training_set.shape
    num_rows_test, num_cols_test = test_set.shape

    distance_matrix = np.zeros((num_rows_test, num_rows_training))

    for i in range(num_rows_test):
        for j in range(num_rows_training):
            D_i_j = distance_matrix[i, j]
            te_Set = test_set[i, :]
            tr_set = training_set[j, :]
            ec_dist = euclidian_distance(te_Set, tr_set)
            distance_matrix[i, j] = ec_dist

    class_vector_guess = np.zeros((num_rows_test, 1))

    for k in range(num_rows_test):
        min_index = np.argmin(np.abs(distance_matrix[k, :]))
        class_vector_guess[k] = class_vector_training_set[min_index]

    return class_vector_guess


def euclidian_distance(x, y):
    x = np.asmatrix(x)
    y = np.asmatrix(y)

    diff_x_y = x - y

    diff_x_y_transpose = diff_x_y.transpose()

    d = diff_x_y.dot(diff_x_y_transpose)

    return d[-1][-1]


def confusion_matrix(guessed_class_vector, true_class_vector, number_of_classes):
    """
    Lager confusion matrix
    c: antall klasser
    """

    number_of_samples, cols = guessed_class_vector.shape
    confusion = np.zeros((number_of_classes, number_of_classes))

    for n in range(number_of_samples):
        true_idx = true_class_vector[n, 0]
        guess_idx = guessed_class_vector[n, 0]

        confusion[int(true_idx), int(guess_idx)] += 1

    return confusion


def error_rate(guessed_class_vector, true_class_vector):
    number_of_samples, cols = guessed_class_vector.shape
    error_count = 0
    for n in range(number_of_samples):
        true_idx = true_class_vector[n, 0]
        guess_idx = guessed_class_vector[n, 0]

        if true_idx != guess_idx:
            error_count += 1

    error_rate = error_count / number_of_samples
    return error_rate

def find_misclassified_indexes(guessed_class_vector, test_labels):
    number_of_samples, cols = guessed_class_vector.shape

    misclassified_indexes = np.array([])

    for n in range(number_of_samples):
        if guessed_class_vector[n, 0] != test_labels[n, 0]:

            misclassified_indexes = np.append(misclassified_indexes, n)

    return misclassified_indexes




def find_correct_classified_indexes(guessed_class_vector, test_labels):
    number_of_samples, cols = guessed_class_vector.shape

    correctly_classified_indexes = np.array([])

    for n in range(number_of_samples):
        if guessed_class_vector[n, 0] == test_labels[n, 0]:

            correctly_classified_indexes = np.append(correctly_classified_indexes,n)

    return correctly_classified_indexes

