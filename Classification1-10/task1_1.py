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
alpha = 0.002           # Prøv ut flere alphaer

labels = ['setosa', 'versicolor', 'virginica']
# Side 13 i kompendie
classes = np.matrix('1 0 0; 0 1 0; 0 0 1')
    #[1, 0, 0]  # setosa
    #[0, 1, 0],  # versicolor
    #[0, 0, 1]  # virginica





W_init = np.ones((C,D)) # Initialiserer CxD- matrise med bare 0
#W_init[0][0] = 3
#W_init[1][1] = 0.2
#W_init[2][2] = 7



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

def sigmoid(value):
    if -value > np.log(np.finfo(type(value)).max):
        return 0.0
    a = np.exp(-value)
    return 1.0/ (1.0 + a)


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

    g_0k = sigmoid(z_k[0])
    g_1k = sigmoid(z_k[1])
    g_2k = sigmoid(z_k[2])

    g_k = np.matrix([g_0k, g_1k, g_2k])
    a = 1
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



W_curr = W_init
training, test = split()
targer_matrix = init_target_matrix()
num_testing_set, num_cols_testing = training.shape
count = 0
while count < 100:  # Kun for test, må ha flere enn 10 iterasjoner
    #print("W:\t", W_curr)
    mse_value = 0
    mse_grad = np.zeros((C,D))
    for k in range(num_testing_set):
        g_k = discriminant_vector(W_curr, training[k,:])
        g_k = g_k.transpose()
        t_k = np.asmatrix(targer_matrix[:,k])
        t_k = t_k.transpose()
        x_k = np.asmatrix(training[k,:])
        x_k = x_k.transpose()

        mse_grad += np.multiply((g_k-t_k), g_k, (np.ones((3,1)) - g_k)).dot(x_k.transpose())
        mse_value += 0.5 * (g_k - t_k).transpose() * (g_k - t_k)

    W_curr -= alpha*mse_grad
    # Oppdater t_k
    if count % 100 == 0:
        print("MSE = \t", mse_value, "\n")

    count += 1

# Calculate value of g
g = np.zeros((C,num_testing_set))
g = np.asmatrix(g)
for k in range(num_testing_set):
    gk = np.asmatrix(discriminant_vector(W_curr, training[k,:])).transpose()
    print("g:", gk, "\n")
    g[:,k] = gk

#print("g:", gk,"\n")






# Oppgave 1c)