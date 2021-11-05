from typing import Union
from pathlib import Path # type: ignore
import numpy as np
from plotly.subplots import make_subplots # type: ignore
import plotly.graph_objects as go # type: ignore
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

from neuron.core.coder import ClampEncoder, SFDecoder
from neuron.core.params_loader import GENERATIONS, POPULATION_SIZE, TIMESTEP, SAMPLES, t
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.genome import Genome
from neuron.optimizer.neat.core import Neat
from neuron.simulation.levitating_ball import LevitatingBall


class Scenario01:
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
                         visualize: bool = False,
                         scenario: str = "") -> Union[float, None]:

        output_decoder_threshold = 1
        output_base = 9.81
        decoder = SFDecoder(output_base, output_decoder_threshold)
        encoder = ClampEncoder()
        cont = genome.build_phenotype(TIMESTEP)
        if visualize:
            v1 = [0.0] * SAMPLES
            v2 = [0.0] * SAMPLES
            v3 = [0.0] * SAMPLES
            spike_trains = [[], [], [], []]

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
            total_error += abs((x_ref - x)/10.0) + abs(x_dot_ref - x_dot)/10
            ######################
            sensors = encoder.encode(e)
            action = cont.step(sensors, t[i], TIMESTEP)
            f = decoder.decode(*action)  # Controller
            x, x_dot = ball.step(f, t[i], TIMESTEP)  # Model

            if visualize:
                v1[i], v2[i] = x, e
                v3[i] = f
                if sensors[0]:
                    spike_trains[0].append(t[i])
                if sensors[1]:
                    spike_trains[1].append(t[i])

                if action[0]:
                    spike_trains[2].append(t[i])
                if action[1]:
                    spike_trains[3].append(t[i])

                if t_10 == 0 and x >= 0.1 * x_ref:
                    t_10 = t[i]

                if t_90 == 0 and x >= 0.9 * x_ref:
                    t_90 = t[i]

        if visualize:
            fig, ax = plt.subplots(4, 1, sharex='all')

            ax[0].plot(t, v1)
            ax[0].grid()
            ax[0].set_title(scenario)
            ax[0].set_ylabel("x(m)")

            ax[1].plot(t, v2)
            ax[1].grid()
            ax[1].set_ylabel("error")

            ax[2].plot(t, v3)
            ax[2].grid()
            ax[2].set_ylabel("force(N)")

            ax[3].eventplot(spike_trains, color=[0, 0, 0], linelengths=0.4)
            ax[3].set_ylabel("Spike Train")
            ax[3].grid()
            ax[3].set_yticks(np.arange(0, 4, 1))
            if visualize:
                print(f"Rise Time = {t_90-t_10}")
            return fig
        if x == 0.0 and x_dot == 0.0:
            return 10000

        return total_error

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
                                    disturbance_magnitude=disturbance_magnitude,
                                    scenario=self.__class__.__name__)
        plt.xlabel("t(s)")
        plt.savefig(f"{self.file_name}_ST.eps")
        plt.show()
        # fig.write_image(f"{self.file_name}.png")
        # genome.visualize(self.file_name)
