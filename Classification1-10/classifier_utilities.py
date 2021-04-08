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


def pred_to_class(pred):

    """
    "Runder av" vektor til nærmeste klasse
    Placeholder
    """
    max_arg = np.argmax(pred)
    new_pred = np.zeros(pred.shape, dtype=int)
    new_pred[max_arg] = 1
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
    err_num = 0
    rounded_g = matr_to_classes(g_matrix)
    print(rounded_g.dtype)
    print(t_matrix.dtype)
    for n in range(rounded_g[0].size):
        if np.array_equal(rounded_g[:,n], t_matrix[:,n]) is False:
            err_num += 1
    error_rate = err_num/g_matrix.size
    return err_num, error_rate

def create_confusion_matrix(g_matrix, targets):
    """
    Lager confusion matrix
    c: antall klasser
    """
    number_of_classes, number_of_samples = g_matrix.shape
    confusion = np.zeros((number_of_classes,number_of_classes))
    rounded_g = matr_to_classes(g_matrix)

    for col in range(number_of_samples):
        # For hver kolonne i g_matrix, sammenlikn med targets og legg til på indeks (klasse g, klasse t) i confusion matrix
        confusion[np.argmax(targets[:,col]),np.argmax(g_matrix[:,col])] += 1

    return confusion
