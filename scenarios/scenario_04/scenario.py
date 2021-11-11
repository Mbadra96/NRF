from typing import Union
from pathlib import Path  # type: ignore
import numpy as np
from plotly.subplots import make_subplots # type: ignore
import plotly.graph_objects as go # type: ignore

from neuron.core.coder import StepEncoder, MWDecoder
from neuron.core.params_loader import GENERATIONS, POPULATION_SIZE, TIME_STEP, SAMPLES, t
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.genome import Genome
from neuron.optimizer.neat.core import Neat
from neuron.simulation.levitating_ball import LevitatingBall
from scenarios.core import SuperScenario

# TODO: NO WORKING


class Scenario(SuperScenario):
    """
    Scenario 04:
                Task : Ball Levitation
                Encoder : Step
                Decoder : Moving-Window
                Case : Reference Tracking & distributed Error
    """
    def __init__(self) -> None:


        self.file_name = f"{Path().absolute()}/scenarios/scenario_04/scenario_04"


    @staticmethod
    def fitness_function(genome: Genome,
                         ref: float = 1.0,
                         mass: float = 1.0,
                         disturbance_magnitude: float = 0.0,
                         visualize: bool = False) -> Union[float, go.Figure]:
        output_decoder_threshold = 1
        output_base = 9.81
        decoder = MWDecoder(10, output_base, output_decoder_threshold)
        cont = genome.build_phenotype(TIME_STEP)
        if visualize:
            v1 = [0.0] * SAMPLES
            v2 = [0.0] * SAMPLES
            v3 = [0.0] * SAMPLES

        x_ref = ref
        x_dot_ref = 0
        total_error = 0.0
        x = 0
        x_dot = 0
        ball = LevitatingBall(mass, x, x_dot)
        encoder_1 = StepEncoder()
        encoder_2 = StepEncoder()
        # Simulation Loop
        for i in range(SAMPLES):
            e1 = (x_ref - x) 
            e2 = (x_dot_ref - x_dot) + Randomizer.Float(-disturbance_magnitude, disturbance_magnitude)
            total_error += abs((x_ref - x)/10.0) + abs(x_dot_ref - x_dot)/10
            ######################
            e = [*encoder_1.encode(e1),*encoder_2.encode(e2)]
            f = decoder.decode(*cont.step(e, t[i], TIME_STEP))  # Controller
            x, x_dot = ball.step(f, t[i], TIME_STEP)  # Model
            if visualize:
                v1[i], v2[i] = x, x_dot
                v3[i] = f
        if visualize:
            if not fig:
                fig = make_subplots(rows=3, cols=1,
                                    shared_xaxes=True,
                                    vertical_spacing=0.02,
                                    x_title="t(s)")
                                

            fig.add_trace(
                go.Scatter(x=t, y=v1, name = 'x'),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=v2,name='x dot'),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=v3,name='f'),
                row=3, col=1
            )
            fig.update_layout(height=720, width=1080, title_text=scenario)
            fig['layout']['yaxis']['title'] = 'x(m)' 
            fig['layout']['yaxis2']['title'] = 'x dot (m/s)' 
            fig['layout']['yaxis3']['title'] = 'force (N)' 
            return fig
        if x == 0.0 and x_dot == 0.0:
            return 10000
        return total_error


    def run(self) -> None:
        # Set Random seed
        Randomizer.seed(0)

        # init NEAT
        self.neat = Neat(4, 2)

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
            print(f"----- Generation {i+1} -----")
            print(f"Generation {i+1} Best = {self.population.best_fitness}")
            print(f"Generation {i+1} Worst = {self.population.worst_fitness}")

        print("-------------------------------------")
        print(f"No OF Species = {self.population.get_species_size()}")
        fig = go.Figure(data=go.Scatter(x=np.arange(1, len(convergence)+1, 1),y=convergence))
        fig.update_layout(height=720, width=1080, title_text=f"{self.__class__.__name__} Convergence")
        fig.update_xaxes(dtick=1)
        fig['layout']['yaxis']['title'] = 'fitness' 
        fig['layout']['xaxis']['title'] = 'generation' 
        
        fig.show()
        fig.write_image(f"{self.file_name}_convergence_curve.png") 
    
    def visualize_and_save(self,ref:float = 1.0 ,mass: float = 1.0, disturbance_magnitude: float = 0.0):
        genome: Genome = Genome.load(self.file_name)
        fig:go.Figure = self.fitness_function(genome,
                                              ref=ref,
                                              visualize=True,
                                              mass=mass,
                                              disturbance_magnitude=disturbance_magnitude)
        fig.show()
        fig.write_image(f"{self.file_name}.png")
        genome.visualize(self.file_name)



