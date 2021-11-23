from typing import Union
from pathlib import Path  # type: ignore
import numpy as np
from plotly.subplots import make_subplots  # type: ignore
import plotly.graph_objects as go  # type: ignore
import matplotlib.pyplot as plt

from neuron.core.coder import StepEncoder, SFDecoder
from neuron.core.params_loader import GENERATIONS, POPULATION_SIZE, TIME_STEP, SAMPLES, t
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.genome import Genome
from neuron.optimizer.neat.core import Neat
from neuron.simulation.levitating_ball import LevitatingBall
from scenarios.core import SuperScenario


class Scenario(SuperScenario):
    """
    Scenario 01:
                Task : Ball Levitation
                Encoder : Clamp
                Decoder : Step-Forward
                Case : Reference Tracking & Central Error
    """
    def __init__(self) -> None:
        self.file_name = f"{Path().absolute()}/scenarios/scenario_01/scenario_01"

    @staticmethod
    def fitness_function(genome: Genome,
                         ref: float = 1.0,
                         mass: float = 1.0,
                         disturbance_magnitude: float = 0.0,
                         visualize: bool = False, fig=None, ax=None) -> Union[float, None]:

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

        x_ref = ref
        x_dot_ref = 0
        total_error = 0.0
        x = 0
        x_dot = 0
        ball = LevitatingBall(mass, x, x_dot)
        t_10 = 0
        t_90 = 0
        # Simulation Loop

        for i in range(SAMPLES):
            e = (x_ref - x) + (x_dot_ref - x_dot) + Randomizer.Float(-disturbance_magnitude, disturbance_magnitude)
            total_error += abs(x_ref - x) + abs(x_dot_ref - x_dot)
            ######################
            sensors = encoder.encode(e)
            action = cont.step(sensors, t[i], TIME_STEP)
            f = decoder.decode(*action)  # Controller
            x, x_dot = ball.step(f, t[i], TIME_STEP)  # Model

            if visualize:
                v1[i], v2[i], v3[i], v4[i] = x, x_dot, e, f
                if t_10 == 0 and x >= 0.1 * x_ref:
                    t_10 = t[i]

                if t_90 == 0 and x >= 0.9 * x_ref:
                    t_90 = t[i]

        if visualize:
            if not fig and not ax:
                fig, ax = plt.subplots(4, 1, sharex='all')

            plt.sca(ax[0])
            plt.cla()
            ax[0].plot(t, v1)
            ax[0].grid()
            ax[0].set_ylabel("x(m)")

            plt.sca(ax[1])
            plt.cla()
            ax[1].plot(t, v2)
            ax[1].grid()
            ax[1].set_ylabel("x dot (m/s)")

            plt.sca(ax[2])
            plt.cla()
            ax[2].plot(t, v3)
            ax[2].grid()
            ax[2].set_ylabel("error")

            plt.sca(ax[3])
            plt.cla()
            ax[3].plot(t, v4)
            ax[3].grid()
            ax[3].set_ylabel("force(N)")

            print(f"Rise Time = {t_90-t_10}")
            return fig, ax

        # # Added Penalty of not moving
        if x == 0.0 and x_dot == 0.0:
            return 10

        return total_error/SAMPLES

    def run(self) -> None:
        # Set Random seed
        Randomizer.seed(0)

        # init NEAT
        self.neat = Neat(2, 2)

        # Generate Population
        self.population = self.neat.generate_population(POPULATION_SIZE, self.fitness_function)

        # Print Start Message
        print(f"Starting Neat with population of {POPULATION_SIZE} for {GENERATIONS} generations")

        convergence: list[float] = []

        for i in range(GENERATIONS):

            # update population
            self.population.update(i == GENERATIONS, save_file_name=self.file_name)
            convergence.append(self.population.best_fitness)
            # print updates
            s = f"----- Generation {i+1} -----\n"
            s += f"Generation {i+1} Best = {self.population.best_fitness}\n"
            s += f"Generation {i+1} Worst = {self.population.worst_fitness}\n"
            print(s)
            self.visualize(False)

        print("-------------------------------------")
        print(f"No OF Species = {self.population.get_species_size()}")
        fig = go.Figure(data=go.Scatter(x=np.arange(1, len(convergence)+1, 1),y=convergence))
        fig.update_layout(height=720, width=1080, title_text=f"{self.__class__.__name__} Convergence")
        fig.update_xaxes(dtick=1)
        fig['layout']['yaxis']['title'] = 'fitness'
        fig['layout']['xaxis']['title'] = 'generation'

        fig.show()
        fig.write_image(f"{self.file_name}_convergence_curve.png")

    def visualize_and_save(self, ref: float = 1.0, mass: float = 1.0, disturbance_magnitude: float = 0.0):

        genome: Genome = Genome.load(self.file_name)
        fig = self.fitness_function(genome,
                                    ref=ref,
                                    visualize=True,
                                    mass=mass,
                                    disturbance_magnitude=disturbance_magnitude)
        plt.xlabel("t(s)")
        plt.savefig(f"{self.file_name}_ST.eps")
        plt.savefig(f"{self.file_name}_ST.png")
        plt.show()
