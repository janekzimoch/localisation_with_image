import numpy as np
import matplotlib.pyplot as plt

from utilis.mdn_utilis import sample_from_output

def plot_dataset(x, y, category_assignment, title=None):
    num_categories = category_assignment.max() + 1  # +1 because python indexes from 0
    for i in range(num_categories):
        indexes = np.where(category_assignment == i)[0]
        plt.scatter(x[indexes], y[indexes])
    plt.xlabel('x')
    plt.ylabel('y')
    if title:
        plt.title(title) # + f" , {num_categories} mixture components")
    plt.show()


def plot_y_pred(data, y_pred, title=None):
    x_train, y_train, cluster_train = data['train']
    x_test, y_test, cluster_test = data['test']

    # train
    num_categories = cluster_train.max() + 1  # +1 because python indexes from 0
    for i in range(num_categories):
        indexes = np.where(cluster_train == i)[0]
        plt.scatter(x_train[indexes], y_train[indexes], alpha=0.01)

    # test
    plt.gca().set_prop_cycle(None)
    for i in range(num_categories):
        indexes = np.where(cluster_test == i)[0]
        plt.scatter(x_test[indexes], y_pred[indexes])

    plt.xlabel('x')
    plt.ylabel('y')
    if title:
        plt.title(title)



def get_y_pred(model, x_test):
    y_pred_dist = model.predict(x_test)
    num_comp = int(y_pred_dist.shape[1] / 3)

    y_pred = np.zeros(len(x_test))
    for i in range(len(x_test)):
        y_pred[i] = sample_from_output(y_pred_dist[i], num_comp)

    return y_pred