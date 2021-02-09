# -*- coding: utf-8 -*-
"""CI_HW1.ipynb


# AT The bottom you will find the polynomial Gradient Decent that tests different polynomial degrees
# the first chart is for linear bellow it is polynomial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand
import os


def load_csv():
    df = pd.read_csv("WHO-COVID-19-Iran-data.csv")
    new_infections = df[df.columns[4]].to_numpy()
    x = list(range(len(new_infections)))
    y = new_infections
    return x, y


# load Data from google DRIVE
# cols = ['New_deaths']
# df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Computational Intellignece /WHO-COVID-19-Iran-data.csv')
# new_infections = df[df.columns[4]].to_numpy()
# t = list(range(len(new_infections)))

# x = t
# y = new_infections

# Split Data into training set
# x_train, y_train, x_validation, y_validation = test_validation_split(x, y)


def main():
    x, y = load_csv()
    x_train, y_train, x_validation, y_validation = test_validation_split(x, y)
    w = 0
    b = 0
    alpha = 0.00001
    epochs = 1000000
    w, b = train_linear(x_train, y_train, w, b, alpha, epochs)
    print(w, " ", b)
    plt.plot(x, y, "o")
    yy = np.dot(w, x) + b
    plt.plot(x, yy)
    plt.show()
    alpha2 = 0.01
    epoch2 = 1000000
    w = list()
    COSTS = list()

    # test code for polynomial Degree of 1 to 14 and the best result of minimum COST on validation is ON degree 8 we
    # call the train function 14 times and train every polynomial degree and then save Weights values in arrays of
    # weights COST is then tried and recorded minimum is taken and the best degree is printed as seen bellow

    for i in range(2, 16):
        X = fx(i, x_train)
        X = normalize(X)
        w.append(multivariate_train(X, y_train, alpha2, epoch2, i))
        print("Polynomial Degree :", i - 1, " finished")
    for i in range(2, 16):
        COSTS.append(cost_multi(normalize(fx(i, x_validation)), y_validation, w[i - 2]))
    COSTS = np.asarray(COSTS)
    print(COSTS)
    min_index = np.argmin(COSTS)
    print("the best fit is ", min_index)
    plt.plot(x, y, "o")
    X = normalize(fx(min_index + 2, x))
    yy = np.dot(w[min_index], np.transpose(X))
    plt.plot(x, yy)
    plt.show()
    # @title Improvement of model by increasing Polynomial Degree maximum improvement is 8
    normal_costs = (COSTS - np.min(COSTS)) / (np.max(COSTS) - np.min(COSTS))
    print(normal_costs)
    plt.plot(list(range(14)), normal_costs)


def test_validation_split(x, y):
    x_train = list()
    x_valid = list()
    y_train = list()
    y_valid = list()
    for i in range(len(x)):
        if i % 6 == 0:
            y_valid.append(y[i])
            x_valid.append(x[i])
        else:
            x_train.append(x[i])
            y_train.append(y[i])
    return x_train, y_train, x_valid, y_valid


def update(x, t, w, b, alpha):
    N = len(x)
    y = np.multiply(w, x) + b
    dedw = np.sum(np.multiply((y - t), x))
    dedb = np.sum(y - t)
    w = w - (alpha / float(N)) * dedw
    b = b - (alpha / float(N)) * dedb
    return w, b


def cost(x, t, w, b):
    N = len(x)
    err = 0
    y = np.multiply(w, x) + b
    err = np.sum((y - t) ** 2)
    return err / float(N * 2)


# y = wx  + b
def train_linear(x, t, w, b, alpha, epochs):
    for e in range(epochs):
        w, b = update(x, t, w, b, alpha)
        if e % 100000 == 0:
            print("epoch", e, " cost:", cost(x, t, w, b))
    return w, b


def multivariate_train(x, t, alpha, epochs, deg):
    w = np.random.rand(deg)

    for e in range(epochs):
        w = update_multi(x, t, w, alpha)
    return w


def update_multi(x, t, w, alpha):
    N = len(t)
    dedw = np.zeros(w.shape)
    y = np.dot(w, np.transpose(x))
    dedw = np.dot(y - t, x)
    w = w - (alpha / float(N)) * dedw
    return w


def fx(deg, x):
    dimension = (len(x), deg)
    array = np.zeros(dimension)
    for i in range(len(x)):
        for j in range(deg):
            array[i][j] = x[i] ** j
    return np.asarray(array)


def normalize(arr):
    arr = np.transpose(arr)
    normalized = np.zeros(np.shape(arr))

    normalized[0] = arr[0]
    # print("this is shape",np.shape(arr))
    for i in range(1, len(arr)):
        normalized[i] = (arr[i] - np.min(arr[i])) / (np.max(arr[i]) - np.min(arr[i]))
        # print("this is min : ",np.min(arr[i]))
        # print("this is max : ",np.max(arr[i]))
    return np.transpose(normalized)


def cost_multi(x, t, w):
    N = len(x)
    err = 0
    # for i in range(N):
    # y = w*x[i]+b
    # err += (y - t[i])**2
    y = np.dot(w, np.transpose(x))
    err = np.sum((y - t) ** 2)
    return err / float(N * 2)


if __name__ == '__main__':
    main()
