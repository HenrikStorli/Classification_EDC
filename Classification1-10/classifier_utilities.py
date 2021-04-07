from math_utilities import *

def discriminant_vector(w_matrix, x_vec):
    # np.append(x_vec, 1.0)
    z_k = w_matrix.dot(np.transpose(x_vec))

    g_0k = sigmoid(z_k[0])
    g_1k = sigmoid(z_k[1])
    g_2k = sigmoid(z_k[2])

    g_k = np.matrix([g_0k, g_1k, g_2k])
    a = 1
    return g_k


def full_g_matrix(W, samples):
    number_of_samples, single_sample_size = samples.shape
    g = np.zeros((C, number_of_samples))
    g = np.asmatrix(g)
    for k in range(number_of_samples):
        gk = np.asmatrix(discriminant_vector(W, samples[k, :])).transpose()
        g[:, k] = gk
    return g


def pred_to_class(pred):

    """
    "Runder av" vektor til nærmeste klasse
    Placeholder
    """
    print("pred ", pred, "stop")
    max_arg = np.argmax(pred)
    new_pred = np.zeros(pred.shape, dtype=int)
    new_pred[max_arg] = 1
    print(max_arg)
    return new_pred

def matr_to_classes(matr):
    """
    Kjører pred_to_class på hel matrise
    """
    rounded_matr = np.empty(matr.shape)
    for i in range(matr[0].size):  # W_matrix[0].size = 90
        rounded_matr[:, i] = np.reshape(pred_to_class(matr[:, i]), 3)  # Runder av til nærmeste klasse

    return rounded_matr

#print("pred_to_class test: ", pred_to_class(np.array([0.5, 3, 0.1])))
#print("pred_to_class test: ", pred_to_class(np.array([1000.0, 1e-3, 10**2])))
#print("pred_to_class test: ", pred_to_class(np.array([1, 2, 3])))

def find_error_rate(g_matrix, t_matrix):
    """
    Finner antall forskjeller mellom gk og tk
    ved avrunding.
    errs: matrise med forskjell mellom W, t
    err_num: antall forskjeller
    error_rate:
    """
    length = g_matrix[0].size
    rounded_g = matr_to_classes(g_matrix)
    rounded_t = matr_to_classes(t_matrix)
    print("rounded_g:", rounded_g)
    errs = np.absolute(np.subtract(rounded_g[0:length], rounded_t))  # Elementwise forskjell W og t
    err_num = np.count_nonzero(errs)
    error_rate = err_num/g_matrix.size
    return errs, err_num, error_rate

def create_confusion_matrix(W_matrix, x_vec, targets, c):
    """
    Ikke ferdig
    Lager confusion matrix
    c: antall klasser
    """
    confusion = np.zeros((c,c))
    pred = np.empty(x_vec.shape)
    num_testing_set, cols_testing_set = x_vec.size
    g_vec = np.zeros((C, num_testing_set))
    g_vec = np.asmatrix(g_vec)
    for k in range(num_testing_set):
        gk = np.asmatrix(discriminant_vector(W_matrix, x_vec[k, :])).transpose()
        print("g:", gk, "\n")
        g_vec[:, k] = gk
    rounded_pred = matr_to_classes(g_vec)
    rounded_targets = matr_to_classes(targets)  # For sikkerhets skyld
    print(g_vec.shape)
    #Sammenlikn
    #for col in range(x_vec.size):
        #if pred[col] == g_vec[col]:
            #confusion[]

    return confusion