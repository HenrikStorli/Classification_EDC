from import_utilities import *


def sigmoid(value):
    if -value > np.log(np.finfo(type(value)).max):
        return 0.0
    a = np.exp(-value)
    return 1.0/ (1.0 + a)


def init_target_matrix(vec):
    T = np.zeros((C,vec*C))
    for row in range(C):
        for column in range(vec):
            T[row][column + row*vec] = 1
    return T

def grad_MSE(g, t, x):
    """

    :param g: diskriminantfunksjon
    :param t: targets
    :param x: data til trening
    :return: side 17 i kompendie
    """
    # Denne gir feil pga multiplikasjon (3,0)-array og (3,)-array
    return np.sum([np.multiply((g-t), g, (1-g))]*np.transpose(x)) # np.multiply: elementwise multiplikasjon


