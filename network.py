import numpy as np
import random


class Network(object):

    # exampole sizes = [784, 100, 10]
    def __init__(self, sizes = []):
        self.sizes = sizes
        self.layers_num = len(sizes)
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weight = [np.random.randn(y, x)
                       for x, y in list(zip(sizes[0:], sizes[1:]))]

    def feedforward(self, a):
        for b, w in list(zip(self.bias, self.weight)):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # eta 学习率
    def SGD(self, epochs, mini_batch_size, training_data, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batch_list = [training_data[k:k + mini_batch_size]
                               for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batch_list:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("%d epoch: %d / %d" %
                      (i, self.evaluate(test_data), n_test))
            else:
                print("%d epoch complete" % (i))

    def evaluate(self, test_data):
        res = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(r == y) for r, y in res)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + delta_w for nw,
                       delta_w in list(zip(nabla_w, delta_nabla_w))]
            nabla_b = [nb + delta_b for nb,
                       delta_b in list(zip(nabla_b, delta_nabla_b))]
        # average
        m = len(mini_batch)
        self.weight = [w - (eta/m) * nw for w,
                       nw in list(zip(self.weight, nabla_w))]
        self.bias = [b - (eta/m) * nb for b, nb in list(zip(self.bias, nabla_b))]

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        # 计算整个网络中的z和activations
        activation = x
        activation_list = [x]
        z_list = []
        for b, w in list(zip(self.bias, self.weight)):
            z = np.dot(w, activation) + b
            z_list.append(z)
            activation = sigmoid(z)
            activation_list.append(activation)
        # 先计算最后一层的误差
        delta = cost_drv(activation_list[-1], y) * sigmoid_drv(z_list[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activation_list[-2].transpose())
        # backward
        for l in range(2, self.layers_num):
            sd = sigmoid_drv(z_list[-l])
            delta = np.dot(self.weight[-l + 1].transpose(), delta) * sd
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activation_list[-l-1].transpose())
        return (nabla_w, nabla_b)


# 损失函数关于激活值的一阶偏导
def cost_drv(a, y):
    return a - y

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# 激活函数一阶导数
def sigmoid_drv(x):
    return sigmoid(x)*(1.0 - sigmoid(x))
