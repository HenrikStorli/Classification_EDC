from classifier_utilities import *

# Oppgave 1 (b)


print(init_target_matrix(training_1b))

W_curr = W_init
training, test = split()
target_matrix = init_target_matrix(training_1b)
num_training_set, num_cols_training = training.shape
g = np.zeros((C, num_training_set))

iters = 1000  # For plotting
mses = []  # For plotting
x_axis = np.arange(start=0, stop=iters, step=100)  # For plotting

count = 0
while count < iters:
    g = full_g_matrix(W_curr, training)
    mse_grad = grad_MSE(target_matrix, g, training)
    W_curr -= alpha * mse_grad

    mse_value = MSE(target_matrix, g)
    if count % 100 == 0:
        print("MSE = \t", mse_value, "\n")
        mses.append(mse_value[-1, -1])
    count += 1

# Calculate value of g
g = full_g_matrix(W_curr, training)
print(g)

# print("g:", gk,"\n")

print("MSE: ", mse_value)
# print(mse_value.shape)


# Plotter MSE som funksjon av antall iterasjoner

plt.plot(x_axis, mses)
plt.xlabel("Iterasjoner")
plt.ylabel("Mean square error")
plt.show()

# Oppgave 1c)


target_matrix_test = init_target_matrix(test_1b)
print("Test targets: ", target_matrix_test)

error_matrix, error_count, error_rate = find_error_rate(g, target_matrix)
print("Errors training: ", error_matrix, '\n', error_count, '\n', round(error_rate * 100, 1), '% \n')

error_matrix_test, error_count_test, error_rate_test = find_error_rate(g[0:], target_matrix_test)
print("Errors test: ", error_matrix_test, '\n', error_count_test, '\n', round(error_rate_test * 100, 1), '% \n')

# Confusion matrix
