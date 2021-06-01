import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Visualize PCA
def PCA_demo(X,K):
    fig, ax = plt.subplots()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    #B1: Tìm điểm nằm ở giữa:
    X_mean=X.mean(axis=0)

    ax.scatter(X_mean[ 0], X_mean[1])
    ax.scatter(X[:, 0], X[:, 1])
    
    #B2 Dời tập các điểm về gốc tọa độ:
    X_hat=X-X_mean
    ax.scatter(X_hat[:, 0], X_hat[:, 1])
    
    #B3: Tìm phương sai
    S=np.dot(X_hat.T,X_hat)/len(X_hat.T[0])
    
    #B4: Tìm trị riêng
    lamb,U=np.linalg.eig(S)
    ind=np.argsort(lamb[::-1])
    U=U[:,ind]

    #B5: lấy k vector riêng
    U_k=U[:,:K]
    
    #B6+7: Chiếu lên vector riêng để lấy kết quả
    Z=np.dot(U_k.T,X_hat.T)
    ax.plot((0,U_k[0]*10), (0,U_k[1]*10))
    ax.plot((0,U[1][0]*10), (0,U[1][1]*10))
    plt.show()
    return U_k,Z,X_mean
#Visualize decode
def decode_demo(U_k,Z,X_mean):
    X_star=np.dot(U_k,Z)+X_mean.reshape(-1,1)
    fig, ax = plt.subplots()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.scatter(X_star.T[:, 0], X_star.T[:, 1])
    ax.scatter(X[:, 0], X[:, 1])
    plt.show()
    return X_star
#init

X=np.array([[1,23,34,49,60,57],[4,54,43,67,70,69]]).T

U_k,Z_k,X_mean = PCA_demo(X,1)
X_star= decode_demo(U_k,Z_k,X_mean)