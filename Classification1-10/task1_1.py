import numpy as np
import matplotlib.pyplot as plt
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

def init_target_matrix(vec):
    T = np.zeros((C,vec*C))
    for row in range(C):
        for column in range(vec):
            T[row][column + row*vec] = 1
    return T

print(init_target_matrix(training_1b))
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




W_curr = W_init
training, test = split()
targer_matrix = init_target_matrix(training_1b)
num_testing_set, num_cols_testing = training.shape

iters = 1000            # For plotting
mses = [] # For plotting
x_axis = np.arange(start=0, stop=iters, step=100)    # For plotting

count = 0
while count < iters:  # Kun for test, må ha flere enn 10 iterasjoner
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
        mses.append(mse_value[-1,-1])
    count += 1

# Calculate value of g
g = np.zeros((C,num_testing_set))
g = np.asmatrix(g)
for k in range(num_testing_set):
    gk = np.asmatrix(discriminant_vector(W_curr, training[k,:])).transpose()
    print("g:", gk, "\n")
    g[:,k] = gk

print(g)

#print("g:", gk,"\n")

print("MSE: ", mse_value)
#print(mse_value.shape)


# Plotter MSE som funksjon av antall iterasjoner

plt.plot(x_axis, mses)
plt.xlabel("Iterasjoner")
plt.ylabel("Mean square error")
plt.show()



# Oppgave 1c)
def find_error_rate(g_matrix, t_matrix):
    """
    Finner antall forskjeller mellom gk og tk
    ved avrunding.
    errs: matrise med forskjell mellom W, t
    err_num: antall forskjeller
    error_rate:
    """
    rounded_g = matr_to_classes(g_matrix)
    rounded_t = matr_to_classes(t_matrix)
    print("rounded_g:", rounded_g)
    errs = np.absolute(np.subtract(rounded_g[0:rounded_t[;,0].size], rounded_t))  # Elementwise forskjell W og t
    err_num = np.count_nonzero(errs)
    error_rate = err_num/g_matrix.size
    return errs, err_num, error_rate

target_matrix_test = init_target_matrix(test_1b)
print("Test targets: ", target_matrix_test)

error_matrix, error_count, error_rate = find_error_rate(g, targer_matrix)
print("Errors training: ", error_matrix, '\n', error_count, '\n', round(error_rate*100,1), '% \n')

error_matrix_test, error_count_test, error_rate_test = find_error_rate(g[0:], target_matrix_test)
print("Errors test: ", error_matrix_test, '\n', error_count_test, '\n', round(error_rate_test*100,1), '% \n')


# Confusion matrix

def create_confusion_matrix(W_matrix, x_vec, targets, c):
    """
    Ikke ferdig
    Lager confusion matrix
    c: antall klasser
    """
    confusion = np.zeros((c,c))
    pred = np.empty(x_vec.shape)

    g_vec = np.zeros((C, num_testing_set))
    g_vec = np.asmatrix(g_vec)
    for k in range(num_testing_set):
        gk = np.asmatrix(discriminant_vector(W_curr, test[k, :])).transpose()
        print("g:", gk, "\n")
        g[:, k] = gk
    rounded_pred = matr_to_classes(g_vec)
    rounded_targets = matr_to_classes(targets)  # For sikkerhets skyld
    print(g_vec.shape)
    #Sammenlikn
    #for col in range(x_vec.size):
        #if pred[col] == g_vec[col]:
            #confusion[]

    return confusion

