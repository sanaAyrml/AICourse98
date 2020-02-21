import random

import matplotlib.pyplot as plt


def random_point():
    x0, y0 = random.uniform(-1, 1), random.uniform(-1, 1)
    return (x0, y0)


class Dataset:
    def target_func(self, p):
        if self.target_a * p[0] + self.target_b > p[1]:
            return -1
        else:
            return 1

    def __init__(self, num_points):
        p0 = random_point()
        p1 = random_point()
        self.target_a = (p1[1] - p0[1]) / (p1[0] - p0[0])
        self.target_b = p0[1] - self.target_a * p0[0]

        self.xs = []
        self.ys = []
        for i in range(num_points):
            xn = random_point()
            self.xs.append(xn)
            self.ys.append(self.target_func(xn))


class Perceptron:

    def __init__(self, dataset):
        self.dataset = dataset
        self.weights = []
        self.weights.append([0,0,0])
        self.iter = 0
        # YOUR CODE

    def fit(self):
        true = 0
        while True:
            # print(true)
            for i in range(len(self.dataset.xs)):
                check = 0
                # print(self.weights[len(self.weights)-1])
                if (self.dataset.xs[i][0] * self.weights[len(self.weights)-1][0] + self.dataset.xs[i][1] * self.weights[len(self.weights)-1][1] + self.weights[len(self.weights)-1][2]) > 0:
                    if self.dataset.ys[i] == -1:
                        check = 1
                else:
                    if self.dataset.ys[i] == 1:
                        check = 1
                # print(check)
                if check:
                    true = 0
                    self.iter += 1
                    weight = [self.weights[len(self.weights)-1][0] + self.dataset.ys[i] * self.dataset.xs[i][0],
                              self.weights[len(self.weights)-1][1] + self.dataset.ys[i] * self.dataset.xs[i][1],
                              self.weights[len(self.weights)-1][2] + self.dataset.ys[i]]
                    self.weights.append(weight)
                else:
                    true += 1
                if true == len(self.dataset.xs):
                    break
            if true == len(self.dataset.xs):
                break
        return

    def plot(self):
        # YOUR CODE
        x0p = []
        x1p = []
        x0m = []
        x1m = []
        for i in range(len(self.dataset.xs)):
            if self.dataset.ys[i] == -1:
                x0m.append( self.dataset.xs[i][0])
                x1m.append(self.dataset.xs[i][1])
            else:
                x0p.append(self.dataset.xs[i][0])
                x1p.append(self.dataset.xs[i][1])
        plt.plot()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.scatter(x0m, x1m, color="red")
        plt.scatter(x0p, x1p, color="blue")
        plt.plot([-1, 1],
                 [-self.dataset.target_a + self.dataset.target_b, self.dataset.target_a + self.dataset.target_b],
                 label="f(X)")
        l = len(self.weights) - 1
        plt.plot([-1, 1], [(-1 * self.weights[l][0] - self.weights[l][2]) / self.weights[l][1],
                           (1 * self.weights[l][0] - self.weights[l][2]) / self.weights[l][1]],label="Perceptron")
        plt.show()

        pass

def find_prceptron(data_set_size):
    test = [random.uniform(-1, 1) for _ in range(1000)]
    count = 0
    prob = 0
    for i in range(5):
        perceptron = Perceptron(Dataset(data_set_size))
        perceptron.fit()
        l = len(perceptron.weights) - 1
        # print(percept.weights[l])
        count += perceptron.iter
        if perceptron.weights[l][1] != 0:
            # perceptron.plot()
            for i in test:
                if abs((i * perceptron.weights[l][0] - perceptron.weights[l][2]) / perceptron.weights[l][1]-perceptron.dataset.target_a * i + perceptron.dataset.target_b) < 0.00001:
                    prob += 1
    return count/5, 1 - prob/ 5

print(find_prceptron(10))
print(find_prceptron(100))
