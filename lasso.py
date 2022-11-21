import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
iris = load_iris()
X = iris.data
y = iris.target
X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)  #   第一列是b后面是w
X = np.concatenate((X, np.random.rand(X.shape[0], 4)), axis=1)    # 在后面添加4列无关变量

def init(dims):
    return np.zeros((dims))

def l1_loss(X, y, w, mylambda):
    y_hat = np.dot(X, w)
    loss = np.sum(np.power(y_hat-y, 2)) / X.shape[0] + np.sum(mylambda * abs(w))
    return loss, y_hat

def lasso(X, y, mylambda=0, max_epochs=5000, tolerance=1e-4):
    w = init(X.shape[1])
    for epoch in range(max_epochs):
        epoch += 1
        done = True
        for i in range(len(w)):
            tmp = w[i]
            w[i] = opt(X, y, w, i, mylambda)
            if (np.abs(tmp-w[i]) > tolerance):
                done = False
        if done:
            break
    return w
def linear(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)      
def opt(X, y, w, i, mylambda):
    level_2 = 0
    level_1 = 0 # level_1为x一次项系数的一半
    for j in range(X.shape[0]):
        x_i = X[j][i]
        other = np.sum([X[j][k] * w[k] for k in range(len(w))]) - X[j][i] * w[i] -y[j] # loss[j] = (x_i*w[i] + other)**2
        level_2 += np.power(x_i, 2)
        level_1 += x_i * other 
    return argmin(level_2, level_1, mylambda)

def argmin(a, b, mylambda): #   求 ax**2 + 2bx + c + mylambda|x|的最小值点
    w1 = - (2 * b + mylambda) / (2 * a) # w>0
    w2 = - (2 * b - mylambda) / (2 * a) # w<0
    if (w1 > 0):
        return w1
    if (w2 < 0):
        return w2
    return 0
# print('线性模型解得的w是：',linear(X, y))
# print('lasso解得的w是：',lasso(X,y))
def get_lambdas():
    lambdas = []
    i = 10
    while (i>=0.001):
        lambdas.append(i)
        if (i>1):
            i -= 0.5
        elif (i>0.1):
            i -= 0.05
        elif (i>0.01):
            i -= 0.005
        else:
            i -= 0.0005
    lambdas.append(i)
    return lambdas
def draw(X, y):
    lambdas = get_lambdas()
    coefs = []
    for lambda_ in lambdas:
        print(lambda_)
        coefs.append(lasso(X, y, lambda_))
    plt.figure()
    plt.title('Curve of Weight Coefficients Changing with Lambda')
    plt.semilogx(lambdas,coefs,'-', label=['b', 'w1', 'w2', 'w3', 'w4', 'noise1', 'noise2', 'noise3', 'noise4'])
    plt.legend(loc='right')
    plt.xlabel('Lamda')
    plt.ylabel('Coefficient')
    plt.savefig(r'lasso.png')
draw(X, y)

def draw_ridge(X, y):
    lambdas = get_lambdas()
    coefs = []
    for lambda_ in lambdas:
        print(lambda_)
        clf = Ridge(alpha=lambda_)
        clf.fit(X, y)
        coefs.append(clf.coef_)
    plt.figure()
    plt.title('Curve of Weight Coefficients Changing with Lambda(Ridge)')
    plt.semilogx(lambdas,coefs,'-', label=['b', 'w1', 'w2', 'w3', 'w4', 'noise1', 'noise2', 'noise3', 'noise4'])
    plt.legend(loc='right')
    plt.xlabel('Lamda')
    plt.ylabel('Coefficient')
    plt.savefig(r'ridge.png')
draw_ridge(X, y)