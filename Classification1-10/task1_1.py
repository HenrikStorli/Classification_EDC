from classifier_utilities import *

# Oppgave 1 (b)


print(init_target_matrix(training_1b))


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


target_matrix_test = init_target_matrix(test_1b)
print("Test targets: ", target_matrix_test)

error_matrix, error_count, error_rate = find_error_rate(g, targer_matrix)
print("Errors training: ", error_matrix, '\n', error_count, '\n', round(error_rate*100,1), '% \n')

error_matrix_test, error_count_test, error_rate_test = find_error_rate(g[0:], target_matrix_test)
print("Errors test: ", error_matrix_test, '\n', error_count_test, '\n', round(error_rate_test*100,1), '% \n')


# Confusion matrix



