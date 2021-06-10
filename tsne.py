import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import gzip

def cal_euclid(X):
    #Tính khoảng cách Euclid giữa các điểm
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return -D

def cal_similarity(X,  zero_index=None):
    e_x = np.exp(X)
    # Đặt xác suất đường chéo = 0 (vì ko tính xs chính nó)
    if zero_index is None:
        np.fill_diagonal(e_x, 0.)
    else:
        e_x[:, zero_index] = 0.

    # Thêm 1 số nhỏ để khi tính log nó sẽ không ra vô cực
    e_x = e_x + 1e-8  # numerical stability
    return e_x / e_x.sum(axis=1).reshape([-1, 1])


def calc_similarity_matrix(distances, sigmas, zero_index=None):
    #Tính sigma
    two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
    return cal_similarity(distances / two_sig_sq, zero_index=zero_index)

def binary_search(eval_fn, target, tol=1e-10, max_iter=10000, lower=1e-20, upper=1000.):
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess


def calc_perplexity(distances, sigmas, zero_index):
    cal_matrix=calc_similarity_matrix(distances, sigmas, zero_index)
    entropy = -np.sum(cal_matrix * np.log2(cal_matrix), 1)
    perplexity = 2 ** entropy
    return perplexity

def find_optimal_sigmas(distances, target_perplexity):
    sigmas = []
    # For each row of the matrix (each point in our dataset)
    print(distances.shape)
    for i in range(distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: \
            calc_perplexity(distances[i:i+1, :], np.array(sigma), i)
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)

def p_joint(X, target_perplexity):
    
    # Get the negative euclidian distances matrix for our data
    distances = cal_euclid(X)
    # Find optimal sigma for each row of this distances matrix
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_similarity_matrix(distances, sigmas)
    # Go from conditional to joint probabilities matrix

    P = (p_conditional + p_conditional.T) / (2. * p_conditional.shape[0])
    return P


def q_tsne(Y):
    distances = cal_euclid(Y)
    inv_distances = np.power(1. - distances, -1)
    np.fill_diagonal(inv_distances, 0.)
    return inv_distances / np.sum(inv_distances), inv_distances


def tsne_grad(P, Q, Y, distances):
    pq_diff = P - Q  # NxN matrix
    pq_expanded = np.expand_dims(pq_diff, 2)  # NxNx1
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  # NxNx2
    # Expand our distances matrix so can multiply by y_diffs
    distances_expanded = np.expand_dims(distances, 2)  # NxNx1
    # Weight this (NxNx2) by distances matrix (NxNx1)
    y_diffs_wt = y_diffs * distances_expanded  # NxNx2
    grad = 4. * (pq_expanded * y_diffs_wt).sum(1)  # Nx2
    return grad

def cal_KL(P,Q):
    log_P=np.log2(P)
    log_Q=np.log2(Q)
    print(Q)
    print(log_Q.shape)
    np.fill_diagonal(log_P, 0.)
    np.fill_diagonal(log_Q, 0.)
    return np.sum(P*(log_P-log_Q))


def estimate_sne(n_components,X, y, PERPLEXITY, random_state, num_iters, learning_rate,momentum):
    P = p_joint(X, PERPLEXITY)

    # Khởi tạo data Y ngẫu nhiên
    Y = np.random.RandomState(random_state).normal(0., 0.0001, [X.shape[0], n_components])
    #print(Y)
    # Initialise past values (used for momentum)
    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

    #Bắt đầu tính gradient descent
    for i in range(num_iters):
        #print(i)
        # Get Q and distances (distances only used for t-SNE)
        Q, distances = q_tsne(Y)
        # Estimate gradients with respect to Y
        grads = tsne_grad(P, Q, Y, distances)
        #print(grads)
        # Update Y
        Y = Y - learning_rate * grads
        if momentum:  # Add momentum
            Y += momentum * (Y_m1 - Y_m2)
            # Update previous Y's for momentum
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()
        #print(cal_KL(P,Q))    
    Q,_=q_tsne(Y)
    kl_divergence=cal_KL(P,Q)
    return Y,kl_divergence


    
# Load the dataset
f = gzip.open('..\input\mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
_, _, test_set = u.load()
f.close()

print(test_set[0].shape)
X=test_set[0][:200,:]
y=test_set[1][:200]
colors = plt.cm.rainbow(np.linspace(0,1,10))


PERPLEXITY=30
Y,kl_divergence=estimate_sne(2,X,y,PERPLEXITY,1,1000,300,True)
plt.scatter(Y[:,0],Y[:,1], c=y, cmap=colors)
plt.show()

print(Y)
print('kl_divergence',kl_divergence)
