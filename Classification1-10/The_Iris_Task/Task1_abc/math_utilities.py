from import_utilities import *
import scipy.stats as stats

def t_test(feature):
    """
    Calculates average of t statistics and p values for
    given feature for all iris classes
    Large t- value -> large difference between 2 sets
    :param feature: index of feature to be compared
    :return: p value
    """
    t_set_versi, p_set_versi  = stats.ttest_ind(setosa[:,feature], versicolor[:,feature])
    t_set_virg, p_set_virg  = stats.ttest_ind(setosa[:,feature], virginica[:,feature])
    t_versi_virg, p_versi_virg = stats.ttest_ind(versicolor[:,feature], virginica[:,feature])

    t_avg = np.average([t_set_versi, t_set_virg, t_versi_virg])
    p_avg = np.average([p_set_versi, p_set_virg, p_versi_virg])

    return t_avg, p_avg


def most_overlapping_feature():
    """
    Compare average t and p values
    """
    t_averages = []
    p_averages = []
    for i in range(num_features):
        t_averages.append(t_test(i)[0])
        p_averages.append(t_test(i)[1])
    t_min_index = np.argmin(np.abs(t_averages))
    p_min_index = np.argmin(np.abs(p_averages))

    return t_min_index, p_min_index


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
    return mse_value[-1,-1]

def grad_MSE(D_grad, target_matrix, g, samples):
    mse_grad = np.zeros((C, D_grad))
    number_of_samples, single_sample_size  = samples.shape
    for k in range(number_of_samples):
        g_k = np.asmatrix(g[:,k])
        t_k = np.asmatrix(target_matrix[:, k])
        t_k = t_k.transpose()
        x_k = np.asmatrix(samples[k, :])
        x_k = x_k.transpose()

        mse_grad += np.multiply((g_k - t_k), g_k, (np.ones((3, 1)) - g_k)).dot(x_k.transpose())
    return mse_grad


