from math import exp
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from neuron.core.params_loader import REFRACTORY_PERIOD, TIME_STEP, TN, C_SYN, R


class NeuroController:

    def __init__(self, connection, inputs, outputs, dt) -> None:
        self.size = len(connection)
        self.input_size = len(inputs)
        self.output_size = len(outputs)

        self.v = np.zeros([self.size, 1])
        self.s = np.zeros([self.size, 1])

        self.refractory_counter = np.zeros([self.size, 1])
        self.refractory_counter_reset = int(REFRACTORY_PERIOD / TIME_STEP) - 1

        self.weight_matrix = np.array(connection).T
        self.input_matrix = np.zeros([self.size, len(inputs)])
        self.output_matrix = np.zeros([len(outputs), self.size])

        # set input matrices
        for i, input_index in enumerate(inputs):
            self.input_matrix[input_index][i] = 1

        # set output matrices
        for i, output_index in enumerate(outputs):
            self.output_matrix[i][output_index] = 1

    def step(self, I, t, dt):
        i = C_SYN * (self.input_matrix @ np.array(I).reshape([self.input_size, 1]) + self.weight_matrix @ self.s)

        self.v = (self.v + (TIME_STEP/TN)*((-self.v) + (R * i))) * np.where(self.refractory_counter == 0, 1, 0)

        self.s = np.where(self.v >= 1, 1, 0)

        self.refractory_counter[self.refractory_counter > 0] -= 1
        self.refractory_counter = np.where(self.v >= 1, self.refractory_counter_reset, self.refractory_counter)

        self.v[self.v < 0] = 0
        self.v[self.v > 1] = 0

        return (self.output_matrix @ self.s).reshape([self.output_size, ]).tolist()
