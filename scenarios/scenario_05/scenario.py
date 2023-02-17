from typing import Union, Any
from pathlib import Path  # type: ignore
from math import inf
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
    Scenario 05:
                Task : Bi-Copter
                Encoder : Step
                Decoder : Step-Forward
                Case : Reference Tracking & Central Error
    """

    def __init__(self) -> None:
        # Set Random seed
        Randomizer.seed(0)

        neat = Neat(2, 4)
        file_name = f"{Path().absolute()}/scenarios/scenario_05/scenario_05"
        super().__init__(file_name=file_name, neat=neat)

    @staticmethod
    def fitness_function(genome: Genome, visualize: bool = False, f_fig=None, f_ax=None, *args, **kwargs) \
            -> Union[float, Any, None]:
        output_decoder_threshold = 0.01
        output_base = 2
        decoder_1 = SFDecoder(output_base, output_decoder_threshold)
        decoder_2 = SFDecoder(output_base, output_decoder_threshold)
        cont = genome.build_phenotype(TIME_STEP)
        theta_ref = 0.0
        theta_dot_ref = 0.0
        total_error = 0.0
        theta = 1.0
        theta_dot = 0.0
        m = kwargs['m'] if ('m' in kwargs) else 1
        copter = BiCopter(theta=theta, m=m)
        disturbance_magnitude = kwargs['disturbance_magnitude'] if ('disturbance_magnitude' in kwargs) else 0
        noise_magnitude = kwargs['noise_magnitude'] if ('noise_magnitude' in kwargs) else 0
        if visualize:
            v1 = [0.0] * SAMPLES
            v2 = [0.0] * SAMPLES
            v3 = [0.0] * SAMPLES
            v4 = [0.0] * SAMPLES
        encoder = StepEncoder()
        t_10 = 0
        t_90 = 0
        # Simulation Loop
        for i in range(SAMPLES):
            e = (theta - theta_ref) + (theta_dot - theta_dot_ref) + Randomizer.Float(-noise_magnitude,
                                                                                     noise_magnitude)
            ###########################
            out = cont.step(encoder.encode(e), t[i], TIME_STEP)
            w1 = decoder_1.decode(out[0], out[1]) + Randomizer.Float(-disturbance_magnitude, disturbance_magnitude)  # Controller
            w2 = decoder_2.decode(out[2], out[3]) + Randomizer.Float(-disturbance_magnitude, disturbance_magnitude)  # Controller
            total_error += abs(e)
            theta, theta_dot = copter.step(w1, w2, t[i], TIME_STEP)  # Model

            if visualize:
                v1[i], v2[i], v3[i], v4[i] = theta, theta_dot, e, (w1, w2)
                if t_10 == 0 and theta <= 0.9:
                    t_10 = t[i]

                if t_90 == 0 and theta <= 0.1:
                    t_90 = t[i]

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
            print(f"Rise Time = {t_90 - t_10}")
            print(f"Error = {total_error / SAMPLES}")
            return f_fig, f_ax

        if theta == 0 and theta_dot == 0:
            return inf
        return total_error / SAMPLES
