import numpy as np
import math
import scipy.signal as sgn
import sys, os
from pathlib import Path

#Feature vektorer
setosa = np.loadtxt("class_1", dtype='float_', delimiter=',')
versicolor = np.loadtxt("class_2", dtype='float_', delimiter=',')
virginica = np.loadtxt("class_3", dtype='float_', delimiter=',')

labeled_set = np.loadtxt("iris.data", dtype='S', delimiter=',')
labeled_set[:,0:3] = labeled_set[:,0:3].astype('float_')


C = 3       # Antall klasser
D = 5       # Antall features + 1, se figur i kompendium
training_1b = 30    # Antall training
test_1b = 20        # Antall test
alpha = 0.2           # Prøv ut flere alphaer

labels = ['setosa', 'versicolor', 'virginica']
# Side 13 i kompendie
classes = np.matrix('1 0 0; 0 1 0; 0 0 1')
    #[1, 0, 0]  # setosa
    #[0, 1, 0],  # versicolor
    #[0, 0, 1]  # virginica





W_init = np.ones((C,D)) # Initialiserer CxD- matrise med bare 0


#print(setosa)
#print(setosa.dtype)

# Oppgave 1 (a)
def split():

    # "Stabler" x- vektorene oppå hverandre
    training = np.vstack((setosa[0:training_1b,:],
                          versicolor[0:training_1b,:],
                          virginica[0:training_1b,:],
                          ))

    test = np.vstack((setosa[training_1b:,:],
                      versicolor[training_1b:,:],
                      virginica[training_1b:,:],
                      ))
    training_ones = np.ones(training.shape[0])
    test_ones = np.ones(test.shape[0])
    # Legger til en kolonne med bare 1'ere
    training = np.append(training, training_ones.reshape(-1,1), axis=1)
    test = np.append(test,test_ones.reshape(-1,1), axis=1)
    return training, test

print("Training: ", split()[0])
print("Test: ", split()[1])


def sigm(z_ik):
    """
    :param z: Wx_k ( = g_k)
    :return: sigmoid av vektorelement nr i
    """
    return 1/(1 + np.exp(-z_ik))


# Oppgave 1 (b)

def init_target_matrix():
    T = np.zeros((C,training_1b*C))
    for row in range(C):
        for column in range(training_1b):
            T[row][column + row*training_1b] = 1
    return T


def discriminant_vector(w_matrix, x_vec):
    # np.append(x_vec, 1.0)
    z_k = w_matrix.dot(np.transpose(x_vec))

    g_0k = sigm(z_k[0])
    g_1k = sigm(z_k[1])
    g_2k = sigm(z_k[2])

    g_k = np.array([g_0k, g_1k , g_2k])
    return g_k

def grad_MSE(g, t, x):
    """

    :param g: diskriminantfunksjon
    :param t: targets
    :param x: data til trening
    :return: side 17 i kompendie
    """
    # Denne gir feil pga multiplikasjon (3,0)-array og (3,)-array
    return np.sum([np.multiply((g-t), g, (1-g))]*np.transpose(x)) # np.multiply: elementwise multiplikasjon


count = 0
W_curr = W_init
training, test = split()


num_testing_set, num_cols_testing = training.shape
targer_matrix = init_target_matrix()

g_k = 0
while count < 150:  # Kun for test, må ha flere enn 10 iterasjoner
    mse_value = 0
    mse_grad = np.zeros((C,D))
    for k in range(num_testing_set):
        g_k = discriminant_vector(W_curr, training[k,:])
        g_k = np.asmatrix(g_k)
        g_k = g_k.transpose()
        t_k = np.asmatrix(targer_matrix[:,k])
        t_k = t_k.transpose()
        x_k = np.asmatrix(training[k,:])
        x_k = x_k.transpose()
        mse_grad += np.multiply((g_k-t_k), g_k, (np.ones((3,1)) - g_k)).dot(x_k.transpose())

        mse_value += 0.5 * (g_k - t_k).transpose() * (g_k - t_k)

    W_curr -= alpha*mse_grad
    # Oppdater t_k
    print("W:\t", W_curr)
    print("MSE = \t", mse_value, "\n")






# Oppgave 1c)