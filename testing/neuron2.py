from math import exp
import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    def __init__(self):
        self.v = 0
        self.s = 0
        self.R = 10
        self.r = 0
        self.dt = 0.0005
        self.tm = 0.001

    def step(self, i: float) -> None:
        if self.r > 0:
            self.r -= 1
            self.v = 0
            self.s = 0
        else:
            self.v = (exp(-self.dt/self.tm)*self.v) + ((self.R * i*self.dt)/self.tm)
            if self.v >= 1.0:
                # self.v = 0
                self.s = 1
                self.r = 4


def main():
    t = np.arange(0, 0.1, 0.0005)
    n = Neuron()

    v: list[list[float], list[float]] = [[], []]

    for i in range(len(t)):
        n.step(1)
        v[0].append(n.v)
        v[1].append(n.s)

    fig, ax = plt.subplots(2, 1, True)
    ax[0].plot(t, v[0])
    ax[0].grid()

    ax[1].plot(t, v[1])
    ax[1].grid()

    plt.show()


if __name__ == '__main__':
    main()

