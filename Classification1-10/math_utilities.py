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

def MSE(target_matrix, g):
    mse_value = 0
    for k in range(g[0].size):
        gk = np.asmatrix(g[:,k])
        #gk = gk.transpose()
        tk = np.asmatrix(target_matrix[:,k])
        tk = tk.transpose()
        mse_value += 0.5 * (gk - tk).transpose() * (gk - tk)
    return mse_value

def grad_MSE(target_matrix, g, samples):
    mse_grad = np.zeros((C, D))
    number_of_samples, single_sample_size  = samples.shape
    for k in range(number_of_samples):
        g_k = np.asmatrix(g[:,k])
        t_k = np.asmatrix(target_matrix[:, k])
        t_k = t_k.transpose()
        x_k = np.asmatrix(samples[k, :])
        x_k = x_k.transpose()

        mse_grad += np.multiply((g_k - t_k), g_k, (np.ones((3, 1)) - g_k)).dot(x_k.transpose())
    return mse_grad


