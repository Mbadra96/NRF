from typing import Union, Any
from pathlib import Path  # type: ignore
import matplotlib.pyplot as plt
from math import sin, cos, pi

from neuron.core.coder import StepEncoder, SFDecoder
from neuron.core.params_loader import TIME_STEP, SAMPLES, t
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.genome import Genome
from neuron.optimizer.neat.core import Neat
from scenarios.core import SuperScenario


class Scenario(SuperScenario):
    """
    Scenario 15:
                Task : Sine Wave
                Encoder : ٍStep
                Decoder : Bang-Bang with Discrete Low Pass Filter
                Case : Tracking & Central Error
    """
    def __init__(self) -> None:
        # Set Random seed
        Randomizer.seed(0)

        neat = Neat(1, 2)
        file_name = f"{Path().absolute()}/scenarios/scenario_17/scenario_17"
        super().__init__(file_name=file_name, neat=neat)

    @staticmethod
    def fitness_function(genome: Genome, visualize: bool = False, f_fig=None, f_ax=None, *args, **kwargs) \
            -> Union[float, Any, None]:
        output_decoder_threshold = 1
        output_base = 9.81
        decoder = SFDecoder(output_base, output_decoder_threshold)
        encoder = StepEncoder()
        cont = genome.build_phenotype(TIME_STEP)
        if visualize:
            v1 = [0.0] * SAMPLES
            v2 = [0.0] * SAMPLES
            v3 = [0.0] * SAMPLES
            v4 = [0.0] * SAMPLES

        total_error = 0
        x = 0
        x_dot = pi/4
        f_filt = 0
        f_filt_last = 0
        w = pi/4
        # Simulation Loop

        for i in range(SAMPLES):
            e = (sin(w * t[i]) - x) + (w * cos(w * t[i]) - x_dot)
            total_error += abs(e)
            ######################

            action = cont.step([1], t[i], TIME_STEP)

            f = - 1 * action[0] + 1 * action[1]

            f_filt = f_filt + 0.005 * (f - f_filt)  # LOW PASS Filter

            x, x_dot = f_filt, (f_filt - f_filt_last)/TIME_STEP
            f_filt_last = f_filt

            if visualize:
                v1[i], v2[i], v3[i], v4[i] = x, x_dot, e, f_filt


        if visualize:
            if not f_fig and not f_ax:
                fig, ax = plt.subplots(4, 1, sharex='all')

            plt.sca(f_ax[0])
            plt.cla()
            f_ax[0].plot(t, v1)
            f_ax[0].grid()
            f_ax[0].set_ylabel("x(m)")

            plt.sca(f_ax[1])
            plt.cla()
            f_ax[1].plot(t, v2)
            f_ax[1].grid()
            f_ax[1].set_ylabel("x dot (m/s)")

            plt.sca(f_ax[2])
            plt.cla()
            f_ax[2].plot(t, v3)
            f_ax[2].grid()
            f_ax[2].set_ylabel("error")

            plt.sca(f_ax[3])
            plt.cla()
            f_ax[3].plot(t, v4)
            f_ax[3].grid()
            f_ax[3].set_ylabel("force(N)")

            return f_fig, f_ax

        # # Added Penalty of not moving
        if x == 0.0 and x_dot == 0.0:
            return 10

        return total_error/SAMPLES


