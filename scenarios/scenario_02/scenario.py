from typing import Union, Any
from pathlib import Path  # type: ignore
import matplotlib.pyplot as plt

from neuron.core.coder import StepEncoder, SFDecoder
from neuron.core.params_loader import TIME_STEP, SAMPLES, t
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.genome import Genome
from neuron.optimizer.neat.core import Neat
from neuron.simulation.levitating_ball import LevitatingBall
from scenarios.core import SuperScenario


class Scenario(SuperScenario):
    """
    Scenario 02:
                Task : Ball Levitation
                Encoder : Step
                Decoder : Step-Forward
                Case : Reference Tracking & distributed Error
    """

    def __init__(self) -> None:
        Randomizer.seed(0)
        file_name = f"{Path().absolute()}/scenarios/scenario_02/scenario_02"
        neat = Neat(4, 2)

        super().__init__(file_name=file_name, neat=neat)

    @staticmethod
    def fitness_function(genome: Genome, visualize: bool = False, f_fig=None, f_ax=None, *args, **kwargs) -> Union[
        float, Any, None]:
        output_decoder_threshold = 1
        output_base = 9.81
        decoder = SFDecoder(output_base, output_decoder_threshold)
        encoder_1 = StepEncoder()
        encoder_2 = StepEncoder()
        cont = genome.build_phenotype(TIME_STEP)

        if visualize:
            v1 = [0.0] * SAMPLES
            v2 = [0.0] * SAMPLES
            v3 = [0.0] * SAMPLES
            v4 = [0.0] * SAMPLES

        x_ref = kwargs['ref'] if ('ref' in kwargs) else 1
        x_dot_ref = 0
        total_error = 0.0
        x = 0
        x_dot = 0
        mass = kwargs['mass'] if ('mass' in kwargs) else 1
        ball = LevitatingBall(mass, x, x_dot)
        disturbance_magnitude = kwargs['disturbance_magnitude'] if ('disturbance_magnitude' in kwargs) else 0
        t_10 = 0
        t_90 = 0
        # Simulation Loop
        for i in range(SAMPLES):
            e1 = (x_ref - x) + Randomizer.Float(-disturbance_magnitude, disturbance_magnitude)
            e2 = (x_dot_ref - x_dot)
            total_error += abs(e1 + e2)
            ######################
            e = [*encoder_1.encode(e1), *encoder_2.encode(e2)]
            action = cont.step(e, t[i], TIME_STEP)
            f = decoder.decode(*action)  # Controller
            x, x_dot = ball.step(f, t[i], TIME_STEP)  # Model
            if visualize:
                v1[i], v2[i], v3[i], v4[i] = x, x_dot, (e1 + e2), f

                if t_10 == 0 and x >= 0.1 * x_ref:
                    t_10 = t[i]

                if t_90 == 0 and x >= 0.9 * x_ref:
                    t_90 = t[i]

        if visualize:
            if not f_fig and not f_ax:
                f_fig, f_ax = plt.subplots(4, 1, sharex='all')

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

            print(f"Rise Time = {t_90 - t_10}")
            return f_fig, f_ax

        # # Added Penalty of not moving
        if x == 0.0 and x_dot == 0.0:
            return 10

        return total_error / SAMPLES
