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
print(labeled_set[0,1].dtype)

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
    training = np.array(
        setosa[0:training_1b,:],
         versicolor[0:training_1b,:],
         virginica[0:training_1b,:]
    )

    test = np.array(
        setosa[-test_1b:-1,:],
         versicolor[-test_1b:-1,:],
         virginica[-test_1b:-1,:]
    )

    return training, test

def sigm(z_ik):
    """
    :param z: Wx_k ( = g_k)
    :return: sigmoid av vektorelement nr i
    """
    return np.array(1/(1 + np.exp(-z_ik)))


#print('Training: ', training, ' (', training.dtype, ')')
#print('Test: ', test, ' (', test.dtype, ')')


# Oppgave 1 (b)

def discriminant_vector(w_matrix, x_vec):
    z_k = w_matrix*x_vec
    g_k = sigm(z_k)
    return g_k

def grad_MSE(g, t, x):
    """

    :param g: diskriminantfunksjon
    :param t: targets
    :param x: data til trening
    :return: side 17 i kompendie
    """
    return np.sum([np.multiply((g-t), g, (1-g))]*np.transpose(x)) # np.multiply: bitwise multiplikasjon


count = 0
W_curr = W_init
while count < 10:  # Kun for test, må ha flere enn 10 iterasjoner
    g_k = discriminant_vector(W_prev, training)
    gradient = grad_MSE(g_k, )
    W_curr -= alpha*grad_MSE()
    print(W_curr)



# Oppgave 1c)