import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import KFold
from statistics import mean


def load_data():
    train_data = np.genfromtxt('hw1xtr.dat', names=True)
    train_data = [np.float(xi[0]) for xi in train_data]

    train_label = np.genfromtxt('hw1ytr.dat', names=True)
    train_label = [np.float(xi[0]) for xi in train_label]

    test_data = np.genfromtxt('hw1xte.dat', names=True)
    test_data = [np.float(xi[0]) for xi in test_data]

    test_label = np.genfromtxt('hw1yte.dat', names=True)
    test_label = [np.float(xi[0]) for xi in test_label]

    return train_data, train_label, test_data, test_label


def plot_data(train_data, train_label, test_data, test_label):
    plt.title("training data visulization")
    plt.scatter(train_data, train_label, c="#1f77b4")
    plt.savefig("training_data_visulization.png")
    plt.close()
    ###########################
    plt.title("testing data visulization")
    plt.scatter(test_data, test_label, c="#ff7f0e")
    plt.savefig("testing_data_visulization.png")
    plt.close()


def add_feature(train_data, order=1):
    n = len(train_data)
    train_data = np.asarray(train_data)
    new_matrix = np.ones((n, (order + 1)))
    new_matrix[:, 0] = train_data
    if order == 2:
        new_matrix[:, 1] = np.square(train_data)
    if order == 3:
        new_matrix[:, 1] = np.square(train_data)
        new_matrix[:, 2] = np.power(train_data, 3)
    if order == 4:
        new_matrix[:, 1] = np.square(train_data)
        new_matrix[:, 2] = np.power(train_data, 3)
        new_matrix[:, 3] = np.power(train_data, 4)

    return new_matrix


def compute_err(y_pred, label):
    m = len(y_pred)
    err = (2 / m) * np.sum(np.square(np.subtract(y_pred, label)))
    return err


def get_gradient(w, x, y, lam, reg=False):
    y_estimate = np.dot(x, w.T).flatten()
    error = np.subtract(np.asarray(y).flatten(), y_estimate)
    if reg:
        gradient = (-(1.0 / len(x)) * np.dot(error, x) + (lam / len(x) * w))
    else:
        gradient = -(1.0 / len(x)) * np.dot(error, x)
    return gradient


def gradient_descent(train_x, train_y, lam, order=2, num_iter=1000, alpha=0.1,
                     reg=False):
    w = np.random.randn(order)  # random initialization of betas
    alpha = alpha

    # Perform Gradient Descent
    iterations = 1
    while True:
        gradient = get_gradient(w, train_x, train_y, lam, reg=reg)
        new_w = w - alpha * gradient

        # Stopping Condition
        if iterations == num_iter:
            return new_w

        iterations += 1
        w = new_w


def linear_regression(train_x, train_y, lam, order):
    I_hat = np.eye(order + 1)
    I_hat[0, 0] = 0
    temp1 = inv(np.dot(train_x.T, train_x) + (lam * I_hat))
    temp2 = np.dot(train_x.T, train_y)
    w = np.dot(temp1, temp2)
    return w


def plot_regression_line(x, y, y_pred, data="training", order=1):
    plt.scatter(x, y, color="navy")
    if order == 1:
        plt.plot(x, y_pred, color="red")
    else:
        plt.scatter(x, y_pred, color="red", marker="*")
    plt.xlabel('data')
    plt.ylabel('label')
    plt.title("linear regression line on {} data order {}".format(data, order))
    plt.savefig("linear_regression_{}_data_order_{}.png".format(data, order))
    plt.close()


def regression(train, train_data, train_label, test, test_data, test_label,
               lam=0, order=1, reg=False):
    # w = gradient_descent(train_data, train_label, lam, order=order + 1,
    # reg=reg)
    w = linear_regression(train_data, train_label, lam, order)
    train_pred = np.dot(train_data, w.T)
    test_pred = np.dot(test_data, w.T)
    if not reg:
        plot_regression_line(train, train_label, train_pred, data="training",
                             order=order)
        plot_regression_line(test, test_label, test_pred, data="testing",
                             order=order)
        train_err = compute_err(train_pred, train_label)
        test_err = compute_err(test_pred, test_label)
        print("average training error: ", train_err)
        print("average testing error: ", test_err)
    return train_pred, test_pred, w


def plot_regularization(lambd, train_error, test_error):
    lambd = np.log10(lambd)
    plt.scatter(lambd, train_error, color="navy", label="train error")
    plt.scatter(lambd, test_error, color="red", marker="*", label="test error")
    plt.legend()
    plt.savefig("regularization_plot.png")
    plt.close()


def plot_weight(lambd, weight):
    weight = np.asarray(weight)
    lambd = np.log10(lambd)
    plt.scatter(lambd, weight[:, 0], color="navy", label="weight_0")
    plt.scatter(lambd, weight[:, 1], color="red", marker="*", label="weight_1")
    plt.scatter(lambd, weight[:, 2], color="black", marker="^",
                label="weight_2")
    plt.scatter(lambd, weight[:, 3], color="green", marker="+",
                label="weight_3")
    plt.scatter(lambd, weight[:, 4], color="pink", marker="x", label="weight_4")
    plt.legend()
    plt.savefig("weight_lambda.png")
    plt.close()


def regularization_regression(train_data, poly_train, train_label, test_data,
                              poly_test, test_label):
    lambd = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    train_error = []
    test_error = []
    weight = []
    for lam in lambd:
        train_pred, test_pred, w = regression(train_data, poly_train,
                                              train_label, test_data, poly_test,
                                              test_label, lam, order=4,
                                              reg=True)
        train_err = compute_err(train_pred, train_label)
        test_err = compute_err(test_pred, test_label)
        train_error.append(train_err)
        test_error.append(test_err)
        weight.append(w)
        if lam == 0.1:
            plot_regression_line(test_data, test_label, test_pred, data="BestFit_Lambda=0.1", order=4)
    plot_regularization(lambd, train_error, test_error)
    plot_weight(lambd, weight)


def cross_validation_regression(train_data, poly_train, train_label, test_data):
    print("Cross Validation")
    lambd = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    for lam in lambd:
        train_error = []
        val_error = []
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(poly_train):
            X_train, X_val = poly_train[train_index], poly_train[test_index]
            y_train, y_val = np.asarray(train_label)[train_index], np.asarray(
                train_label)[test_index]
            train_pred, val_pred, w = regression(train_data, X_train, y_train,
                                                 test_data, X_val, y_val,
                                                 lam, order=4, reg=True)
            train_error.append(compute_err(train_pred, y_train))
            val_error.append(compute_err(val_pred, y_val))
        print("Lambda {} train error: {} validation error: {}".format(lam, mean(
            train_error), mean(val_error)))


def main():
    print("Loading data...")
    train_data, train_label, test_data, test_label = load_data()
    plot_data(train_data, train_label, test_data, test_label)
    print("--------------------------")
    print("Linear Regression")
    poly_train = add_feature(train_data, order=1)
    poly_test = add_feature(test_data, order=1)
    regression(train_data, poly_train, train_label, test_data, poly_test,
               test_label, order=1)
    print("--------------------------")
    print("Second Order Polynomial Regression")
    poly_train = add_feature(train_data, order=2)
    poly_test = add_feature(test_data, order=2)
    regression(train_data, poly_train, train_label, test_data, poly_test,
               test_label, order=2)
    print("--------------------------")
    print("Third Order Polynomial Regression")
    poly_train = add_feature(train_data, order=3)
    poly_test = add_feature(test_data, order=3)
    regression(train_data, poly_train, train_label, test_data, poly_test,
               test_label, order=3)
    print("--------------------------")
    print("Fourth Order Polynomial Regression")
    poly_train = add_feature(train_data, order=4)
    poly_test = add_feature(test_data, order=4)
    regression(train_data, poly_train, train_label, test_data, poly_test,
               test_label, order=4)
    print("--------------------------")
    print("Fourth Order Polynomial Regression with Regularization")
    poly_train = add_feature(train_data, order=4)
    poly_test = add_feature(test_data, order=4)
    # Question3 parts (a) and (b)
    regularization_regression(train_data, poly_train, train_label, test_data,
                              poly_test, test_label)

    # Question3 parts (c)
    cross_validation_regression(train_data, poly_train, train_label, test_data)
    print("--------------------------")


if __name__ == "__main__":
    main()
