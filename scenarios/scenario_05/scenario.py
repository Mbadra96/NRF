from typing import Union
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
    Scenario 05:
                Task : Bi-Copter
                Encoder : Step
                Decoder : Step-Forward
                Case : Reference Tracking & Central Error
    """

    def __init__(self) -> None:
        self.file_name = f"{Path().absolute()}/scenarios/scenario_05/scenario_05"

    @staticmethod
    def fitness_function(genome: Genome,
                         visualize: bool = False,
                         disturbance_magnitude: float = 0.0) -> Union[float, go.Figure]:
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
        copter = BiCopter(theta=theta)
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
            w1 = decoder_1.decode(out[0], out[1])  # Controller
            w2 = decoder_2.decode(out[2], out[3])  # Controller
            total_error += abs((theta - theta_ref) / 10.0) + (abs(theta_dot - theta_dot_ref) / 10.0)
            theta, theta_dot = copter.step(w1, w2, t[i], TIME_STEP)  # Model

            if visualize:
                v1[i], v2[i], v3[i], v4[i] = theta, theta_dot, e, (w1, w2)

        if visualize:
            fig, ax = plt.subplots(4, 1, sharex='all')

            ax[0].plot(t, v1)
            ax[0].grid()
            ax[0].set_ylabel("theta")

            ax[1].plot(t, v2)
            ax[1].grid()
            ax[1].set_ylabel("theta dot (m/s)")

            ax[2].plot(t, v3)
            ax[2].grid()
            ax[2].set_ylabel("error")
            # ax[2].set_yticks(np.arange(-0.05, 0.09, 0.05))

            ax[3].plot(t, v4)
            ax[3].grid()
            ax[3].set_ylabel("W1 & W2")
            # ax[3].set_yticks(np.arange(7, 14, 1))

        if theta == 0 and theta_dot == 0:
            return inf
        return total_error / SAMPLES

    def run(self) -> None:
        # Set Random seed
        Randomizer.seed(0)

        # init NEAT
        self.neat = Neat(2, 4)

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
            s = f"----- Generation {i + 1} -----\n"
            s += f"Generation {i + 1} Best = {self.population.best_fitness}\n"
            s += f"Generation {i + 1} Worst = {self.population.worst_fitness}\n"
            print(s)

        print("-------------------------------------")
        print(f"No OF Species = {self.population.get_species_size()}")
        fig = go.Figure(data=go.Scatter(x=np.arange(1, len(convergence) + 1, 1), y=convergence))
        fig.update_layout(height=720, width=1080, title_text=f"{self.__class__.__name__} Convergence")
        fig.update_xaxes(dtick=1)
        fig['layout']['yaxis']['title'] = 'fitness'
        fig['layout']['xaxis']['title'] = 'generation'

        fig.show()
        fig.write_image(f"{self.file_name}_convergence_curve.png")

    def visualize_and_save(self):
        genome: Genome = Genome.load(self.file_name)
        self.fitness_function(genome, visualize=True)

        plt.xlabel("t(s)")
        plt.savefig(f"{self.file_name}_ST.eps")
        plt.show()
        genome.visualize(self.file_name)
