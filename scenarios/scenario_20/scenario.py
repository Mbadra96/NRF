from typing import Union, Any
from pathlib import Path  # type: ignore
import matplotlib.pyplot as plt

from neuron.core.coder import StepEncoder
from neuron.core.params_loader import TIME_STEP, SAMPLES, t
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.genome import Genome
from neuron.optimizer.neat.core import Neat
from neuron.simulation.motor import Motor
from scenarios.core import SuperScenario


class Scenario(SuperScenario):
    """
    Scenario 19:
                Task : Motor Control
                Encoder : Step
                Decoder : Bang-Bang-Filter
                Case : Reference Tracking & Central Error
    """
    def __init__(self) -> None:
        # Set Random seed
        Randomizer.seed(0)

        neat = Neat(2, 2)
        file_name = f"{Path().absolute()}/scenarios/scenario_20/scenario_20"
        super().__init__(file_name=file_name, neat=neat)

    @staticmethod
    def fitness_function(genome: Genome, visualize: bool = False, f_fig=None, f_ax=None, *args, **kwargs) -> Union[float, Any, None]:
        encoder = StepEncoder()
        cont = genome.build_phenotype(TIME_STEP)
        if visualize:
            v1 = [0.0] * SAMPLES
            v2 = [0.0] * SAMPLES
            v3 = [0.0] * SAMPLES
        w_ref = 1  # rad/sec
        w = 0
        total_error = 0
        motor = Motor(w_0=w)
        disturbance_magnitude = kwargs['disturbance_magnitude'] if ('disturbance_magnitude' in kwargs) else 0
        t_10 = 0
        t_90 = 0
        f_filt = 0
        # Simulation Loop
        for i in range(SAMPLES):
            e = (w_ref - w) + Randomizer.Float(-disturbance_magnitude, disturbance_magnitude)
            total_error += abs(e)
            ######################
            sensors = encoder.encode(e)
            action = cont.step(sensors, t[i], TIME_STEP)
            f = - 100 * action[0] + 100 * action[1]
            # f_filt = f
            f_filt = f_filt + 0.005 * (f - f_filt)
            w = motor.step(f_filt, t[i], TIME_STEP)  # Model

            if visualize:
                v1[i], v2[i], v3[i] = w, e, f_filt
                if t_10 == 0 and w >= 0.1 * w_ref:
                    t_10 = t[i]

                if t_90 == 0 and w >= 0.9 * w_ref:
                    t_90 = t[i]

        if visualize:
            if not f_fig and not f_ax:
                fig, ax = plt.subplots(3, 1, sharex='all')

            plt.sca(f_ax[0])
            plt.cla()
            f_ax[0].plot(t, v1)
            f_ax[0].grid()
            f_ax[0].set_ylabel("w (rad/sec)")

            plt.sca(f_ax[1])
            plt.cla()
            f_ax[1].plot(t, v2)
            f_ax[1].grid()
            f_ax[1].set_ylabel("error")

            plt.sca(f_ax[2])
            plt.cla()
            f_ax[2].plot(t, v3)
            f_ax[2].grid()
            f_ax[2].set_ylabel("Voltage (V)")
            print(f"Error = {total_error/SAMPLES}")
            print(f"Rise Time = {t_90-t_10}")
            return f_fig, f_ax

        return total_error/SAMPLES


