from math import exp
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from neuron.core.params_loader import REFRACTORY_PERIOD, TIME_STEP, TN

class NeuroController:
    def __init__(self, connection, inputs, outputs, dt) -> None:
        self.size = len(connection)
        self.v = np.zeros([self.size, 1])
        self.s = np.zeros([self.size, 1])
        self.refractory_counter = np.zeros([self.size, 1])
        self.refractory_counter_reset = REFRACTORY_PERIOD/TIME_STEP - 1
        self.tm = 0.001 #0.004745610791
        self.Cv = exp(-TIME_STEP/self.tm)
        self.R = 10 #1/(1-self.Cv)
        self.Ci = self.R*(1-self.Cv)
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

        # assert len(I) == len(self.input_indices), f"Size of inputs should be {len(self.input_indices)} while given {len(I)}"
        
        i = self.input_matrix @ np.array(I).reshape([len(I),1]) + self.weight_matrix @ self.s
        self.v = ((self.Cv * self.v) + (self.Ci * i))*np.where(self.refractory_counter == 0, 1, 0)
        self.s = np.where(self.v >= 1, 1, 0)
        self.refractory_counter[self.refractory_counter > 0] -= 1
        self.refractory_counter = np.where(self.v >= 1, self.refractory_counter_reset, self.refractory_counter)
        self.v[self.v < 0] = 0

        return self.output_matrix @ self.v

