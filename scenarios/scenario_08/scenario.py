from typing import Union
from pathlib import Path # type: ignore
import numpy as np
from math import inf
from plotly.subplots import make_subplots # type: ignore
import plotly.graph_objects as go # type: ignore


from neuron.core.coder import StepEncoder, MWDecoder
from neuron.core.params_loader import GENERATIONS, POPULATION_SIZE, TIME_STEP, SAMPLES, t
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.genome import Genome
from neuron.optimizer.neat.core import Neat
from neuron.simulation.bicopter2 import BiCopter
from scenarios.core import SuperScenario

# TODO: NEEDS TO BE UPDATED


class Scenario(SuperScenario):
    """
    Scenario 08:
                Task : Bi-Copter
                Encoder : Step
                Decoder : Moving-Window
                Case : Reference Tracking & Distributed Error
    """
    def __init__(self) -> None:
        self.file_name = f"{Path().absolute()}/scenarios/scenario_08/scenario_08"

    @staticmethod
    def fitness_function(genome: Genome,
                         visualize: bool = False) -> Union[float, go.Figure]:
        output_decoder_threshold = 0.01
        output_base = 0.5
        decoder_1 = MWDecoder(5, output_base, output_decoder_threshold)
        decoder_2 = MWDecoder(5, output_base, output_decoder_threshold)
        encoder1 = StepEncoder()
        encoder2 = StepEncoder()
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

        # Simulation Loop
        for i in range(SAMPLES):
            e1 = (theta - theta_ref)
            e2 = (theta_dot - theta_dot_ref) #+ Randomizer.Float(-disturbance_magnitude, disturbance_magnitude)
            total_error += abs((theta - theta_ref) / 10.0) + (abs(theta_dot - theta_dot_ref)/10.0)
            ###########################
            out = cont.step([*encoder1.encode(e1), *encoder2.encode(e2)], t[i], TIME_STEP)
            w1 = decoder_1.decode(out[0],out[1])  # Controller
            w2 = decoder_2.decode(out[2],out[3])  # Controller
            theta, theta_dot = copter.step(w1, w2, t[i], TIME_STEP)  # Model

            if visualize:
                v1[i], v2[i] = theta, theta_dot
                v3[i] = w1
                v4[i] = w2

        if visualize:
            if not fig:
                fig = make_subplots(rows=4, cols=1, subplot_titles=("theta", "theta Dot", "w1", "w2"))

            fig.add_trace(
                go.Scatter(x=t, y=v1, name="theta"),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=v2, name="theta dot"),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=v3, name="w1"),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=t, y=v4, name="w2"),
                row=4, col=1
            )
            fig.update_layout(height=720, width=1080, title_text=scenario)

            return fig
        if theta == 0 and theta_dot == 0:
            return inf
        return total_error

    def run(self) -> None:
        # Set Random seed
        Randomizer.seed(0)

        # init NEAT
        self.neat = Neat(4, 4)

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

        print("-------------------------------------")
        print(f"No OF Species = {self.population.get_species_size()}")
        fig = go.Figure(data=go.Scatter(x=np.arange(1, len(convergence)+1, 1), y=convergence))
        fig.update_layout(height=720, width=1080, title_text=f"{self.__class__.__name__} Convergence")
        fig.update_xaxes(dtick=1)
        fig['layout']['yaxis']['title'] = 'fitness'
        fig['layout']['xaxis']['title'] = 'generation'

        fig.show()
        fig.write_image(f"{self.file_name}_convergence_curve.png")

    def visualize_and_save(self):
        genome: Genome = Genome.load(self.file_name)
        fig: go.Figure = self.fitness_function(genome, visualize=True)
        fig.show()
        fig.write_image(f"{self.file_name}.png") 
        genome.visualize(self.file_name)
