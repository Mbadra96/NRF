from math import exp
import matplotlib.pyplot as plt #type: ignore
import numpy as np


class Neuron:
    def __init__(self):
        self.v = np.zeros([2])
        self.s = np.zeros([2])
        # self.R = 10
        self.refractory_counter = np.zeros([2])
        self.refractory_counter_reset = 3 # refractory_period/dt -1
        self.dt = 0.0005
        self.tm = 0.01 #0.004745610791
        self.Cv = exp(-self.dt/self.tm)
        self.R = 1/(1-self.Cv)
        self.Ci = self.R*(1-self.Cv)
        self.weight_matrix = np.array([[0, 0], [1, 0]])

    def step(self, i_ext: list[float]) -> None:

        i = np.array(i_ext) + self.weight_matrix @ self.s

        self.v = ((self.Cv * self.v) + (self.Ci * i))*np.where(self.refractory_counter == 0, 1, 0) 
        self.s = np.where(self.v >= 1, 1, 0) 
        self.refractory_counter[self.refractory_counter > 0] -= 1
        self.refractory_counter = np.where(self.v >= 1, self.refractory_counter_reset, self.refractory_counter)
        self.v[self.v < 0 ] = 0


def main():
    t = np.arange(0, 1, 0.0005)
    n = Neuron()

    v: list[list[float], list[float], list[float]] = [[], [], []]
    from math import sin, pi
    for i in range(len(t)):
        n.step([(sin(10*pi*t[i])+1), 0])
        v[0].append(n.v[0])
        v[1].append((sin(10*pi*t[i])+1)/2)
        if n.s[0] == 1:
            v[2].append(t[i])

    _, ax = plt.subplots(3, 1, sharex='all')
    ax[0].plot(t, v[0])
    ax[0].grid()

    ax[1].plot(t, v[1])
    ax[1].grid()

    ax[2].eventplot(v[2], colors=(0, 1, 0))
    ax[2].grid()

    plt.show()


if __name__ == '__main__':
    main()

