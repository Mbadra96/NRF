from math import exp
import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    def __init__(self):
        self.v = np.zeros([2])
        self.s = np.zeros([2])
        # self.R = 10
        self.r = np.zeros([2])
        self.dt = 0.0005
        self.tm = 0.001
        self.Cv = exp(-self.dt/self.tm)
        self.R = 1/(1-self.Cv)
        self.Ci = self.R*(1-self.Cv)
        self.weight_matrix = np.array([[0, 0], [1, 0]])

    def step(self, i_ext: list[float]) -> None:
        # Minimum Current tm*(1-e^c)/(R dt) as
        # it a lim when v = 1 => 1 = self.v = (self.c * 1) + ((self.R * i*self.dt)/self.tm)

        i = np.array(i_ext) + self.weight_matrix @ self.s

        self.v = ((self.Cv * self.v) + (self.Ci * i))*np.where(self.r == 0, 1, 0)

        self.s = np.where(self.v >= 1, 1, 0)
        self.r[self.r > 0] -= 1
        self.r = np.where(self.v >= 1, 4, self.r)


def main():
    t = np.arange(0, 0.1, 0.0005)
    n = Neuron()

    v: list[list[float], list[float], list[float]] = [[], [], []]

    for i in range(len(t)):
        n.step([0.8, 0])
        v[0].append(n.v[0])
        v[1].append(n.s[0])
        v[2].append(n.s[0]*t[i])

    fig, ax = plt.subplots(3, 1, sharex='all')
    ax[0].plot(t, v[0])
    ax[0].grid()

    ax[1].plot(t, v[1])
    ax[1].grid()

    ax[2].eventplot(v[2], colors=(0, 1, 0))
    ax[2].grid()

    plt.show()


if __name__ == '__main__':
    main()

