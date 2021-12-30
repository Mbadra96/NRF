from typing import Union, Any
from pathlib import Path  # type: ignore
import numpy as np
from math import inf
from plotly.subplots import make_subplots  # type: ignore
import plotly.graph_objects as go  # type: ignore
import matplotlib.pyplot as plt

from neuron.core.coder import StepEncoder, SFDecoder
from neuron.core.params_loader import GENERATIONS, POPULATION_SIZE, TIME_STEP, SAMPLES, t
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.genome import Genome
from neuron.optimizer.neat.core import Neat
from neuron.simulation.bicopter2 import BiCopter
from scenarios.core import SuperScenario


class Scenario(SuperScenario):
    """
    Scenario 18:
                Task : Bi-Copter
                Encoder : Step
                Decoder : Bang Bang Filter
                Case : Reference Tracking & Central Error
    """

    def __init__(self) -> None:
        Randomizer.seed(0)
        file_name = f"{Path().absolute()}/scenarios/scenario_18/scenario_18"
        neat = Neat(2, 4)

        super().__init__(file_name=file_name, neat=neat)

    @staticmethod
    def fitness_function(genome: Genome, visualize: bool = False, f_fig=None, f_ax=None, *args, **kwargs) -> Union[
        float, Any, None]:

        cont = genome.build_phenotype(TIME_STEP)
        disturbance_magnitude = kwargs['disturbance_magnitude'] if ('disturbance_magnitude' in kwargs) else 0
        theta_ref = 0.0
        theta_dot_ref = 0.0
        total_error = 0.0
        theta = 1.0
        theta_dot = 0.0
        copter = BiCopter(theta=theta)

        w_1_filt = 0
        w_2_filt = 0

        if visualize:
            v1 = [0.0] * SAMPLES
            v2 = [0.0] * SAMPLES
            v3 = [0.0] * SAMPLES
            v4 = [0.0] * SAMPLES
        encoder = StepEncoder()
        # Simulation Loop
        for i in range(SAMPLES):
            e = (theta - theta_ref) + (theta_dot - theta_dot_ref) + Randomizer.Float(-disturbance_magnitude,
                                                                                     disturbance_magnitude)
            ###########################
            out = cont.step(encoder.encode(e), t[i], TIME_STEP)

            f1 = - out[0] + out[1]
            f2 = - out[2] + out[3]

            w_1_filt = w_1_filt + 0.005 * (10*f1 - w_1_filt)  # LOW PASS Filter
            w_2_filt = w_2_filt + 0.005 * (10*f2 - w_2_filt)  # LOW PASS Filter

            total_error += abs(e)
            theta, theta_dot = copter.step(w_1_filt, w_2_filt, t[i], TIME_STEP)  # Model

            if visualize:
                v1[i], v2[i], v3[i], v4[i] = theta, theta_dot, e, (w_1_filt, w_2_filt)

        if visualize:
            if not f_fig and not f_ax:
                fig, ax = plt.subplots(4, 1, sharex='all')

            plt.sca(f_ax[0])
            plt.cla()
            f_ax[0].plot(t, v1)
            f_ax[0].grid()
            f_ax[0].set_ylabel("theta")

            plt.sca(f_ax[1])
            plt.cla()
            f_ax[1].plot(t, v2)
            f_ax[1].grid()
            f_ax[1].set_ylabel("theta dot")

            plt.sca(f_ax[2])
            plt.cla()
            f_ax[2].plot(t, v3)
            f_ax[2].grid()
            f_ax[2].set_ylabel("error")

            plt.sca(f_ax[3])
            plt.cla()
            f_ax[3].plot(t, v4)
            f_ax[3].grid()
            f_ax[3].set_ylabel("W1 & W2")

            return f_fig, f_ax

        if theta == 0 and theta_dot == 0:
            return inf
        return total_error / SAMPLES
