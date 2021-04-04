import numpy as np
import math
import scipy.signal as sgn
import sys, os
from pathlib import Path

#Feature vektorer
setosa = np.loadtxt("C:\\Users\\Lokal\\Documents\\ELSYS\\3V\\EDK\\prosjekt\\Classification1-10\\class_iris\\Iris_TTT4275\\class_1", dtype='float_', delimiter=',')
versicolor = np.loadtxt("C:\\Users\\Lokal\\Documents\\ELSYS\\3V\\EDK\\prosjekt\\Classification1-10\\class_iris\\Iris_TTT4275\\class_2", dtype='float_', delimiter=',')
virginica = np.loadtxt("C:\\Users\\Lokal\\Documents\\ELSYS\\3V\\EDK\\prosjekt\\Classification1-10\\class_iris\\Iris_TTT4275\\class_3", dtype='float_', delimiter=',')

labeled_set = np.loadtxt("C:\\Users\\Lokal\\Documents\\ELSYS\\3V\\EDK\\prosjekt\\Classification1-10\\class_iris\\Iris_TTT4275\\iris.data", dtype='S', delimiter=',')
labeled_set[:,0:3] = labeled_set[:,0:3].astype('float_')


C = 3       # Antall klasser
D = 5       # Antall features + 1, se figur i kompendium
training_1b = 30    # Antall training
test_1b = 20        # Antall test
alpha = 1           # Prøv ut flere alphaer

labels = ['setosa', 'versicolor', 'virginica']
# Side 13 i kompendie
classes = np.array(
    [[1, 0, 0],  # setosa
    [0, 1, 0],  # versicolor
    [0, 0, 1]]  # virginica
)

W_init = np.zeros((C,D)) # Initialiserer CxD- matrise med bare 0


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

def discriminant_vector(w_matrix, x_vec):
    np.append(x_vec, 1.0)
    z_k = w_matrix.dot(np.transpose(x_vec))
    g_k = sigm(z_k)
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
t_k = classes[0]    # start på setosa,
while count < 150:  # Kun for test, må ha flere enn 10 iterasjoner
    g_k = 0
    for i in range(training.size):
        g_k = discriminant_vector(W_curr, training[i:,])
    gradient = grad_MSE(g_k, t_k, training)
    W_curr -= alpha*gradient
    # Oppdater t_k
    if count > 50:
        t_k = classes[1]        # versicolor
        if count > 100:
            t_k = classes[2]        # virginica
    print("W:\t", W_curr)





# Oppgave 1c)