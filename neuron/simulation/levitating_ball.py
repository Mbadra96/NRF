from neuron.core.coder import ClampEncoder, SFDecoder, MWDecoder
from neuron.utils.functions import clamp_signal_to_spikes
from neuron.core.params_loader import params
from neuron.utils.units import *
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.genome import Genome
import numpy as np
from plotly.subplots import make_subplots # type: ignore
import plotly.graph_objects as go # type: ignore

TIME = params["Evaluation_Time"] * sec
TIMESTEP = params["Time_Step"] * ms  # dt is 0.5 ms
SAMPLES = int(TIME / TIMESTEP)
t = np.arange(0, TIME, TIMESTEP)


class LevitatingBall:

    INPUT_NEURONS = 2
    OUTPUT_NEURONS = 2

    def __init__(self, mass, x_0, x_dot_0) -> None:
        self.mass = mass
        self.x = x_0
        self.x_dot = x_dot_0

    def step(self, i, t, dt):

        self.x_dot += dt*(i-self.mass * 9.81)/self.mass
        self.x += dt*self.x_dot
        
        if self.x <= 0:
            self.x = 0
            self.x_dot = 0

        elif self.x >= 10:
            self.x = 10
            self.x_dot = 0
            
        return self.x, self.x_dot

    @staticmethod
    def evaluate_genome(genome: Genome, mass: float = 1.0, disturbance_magnitude: float = 0.0) -> float:
        output_decoder_threshold = 1
        output_base = 9.81
        decoder = SFDecoder(output_base, output_decoder_threshold)
        # decoder = MWDecoder(20, output_base, output_decoder_threshold)
        cont = genome.build_phenotype(TIMESTEP)

        x_ref = 2
        x_dot_ref = 0
        total_error = 0.0
        x = 0
        x_dot = 0
        ball = LevitatingBall(mass, x, x_dot)

        # Simulation Loop
        for i in range(SAMPLES):
            e = (x_ref - x) + (x_dot_ref - x_dot) #+ Randomizer.Float(-disturbance_magnitude, disturbance_magnitude)
            total_error += abs((x_ref - x) / 10.0)
            ###########################
            # sensors = [*
            ######################
            f = decoder.decode(*cont.step(clamp_signal_to_spikes(e), t[i], TIMESTEP))  # Controller
            x, x_dot = ball.step(f, t[i], TIMESTEP)  # Model

        return total_error

    @staticmethod
    def evaluate_genome_with_figure(genome: Genome, show: bool = False, mass: float = 1.0, disturbance: bool = False,
                                    disturbance_magnitude=0.0, fig=None) -> go.Figure:
        output_decoder_threshold = 1
        output_base = 9.81
        decoder = SFDecoder(output_base, output_decoder_threshold)
        cont = genome.build_phenotype(TIMESTEP)

        if show:
            v1 = [0.0] * SAMPLES
            v2 = [0.0] * SAMPLES
            v3 = [0.0] * SAMPLES

        x_ref = 2
        x_dot_ref = 0
        total_error = 0.0
        x = 0
        x_dot = 0
        ball = LevitatingBall(mass, x, x_dot)
        encoder = ClampEncoder()
        # Simulation Loop
        for i in range(SAMPLES):
            e = (x_ref - x) + (x_dot_ref - x_dot) + Randomizer.Float(-disturbance_magnitude, disturbance_magnitude)
            total_error += abs((x_ref - x) / 10.0)
            ######################
            F = decoder.decode(*cont.step(encoder.encode(e), t[i], TIMESTEP))  # Controller
            x, x_dot = ball.step(F, t[i], TIMESTEP)  # Model

            if show:
                v1[i], v2[i] = x, x_dot
                v3[i] = F
            else:
                pass

        if show:
            if not fig:
                fig = make_subplots(rows=3, cols=1, subplot_titles=("X", "X Dot", "Force"))

            fig.add_trace(
                go.Scatter(x=t, y=v1, name=f"X D={disturbance_magnitude:.2f}"),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=v2, name=f"X Dot D={disturbance_magnitude:.2f}"),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=v3, name=f"Force D={disturbance_magnitude:.2f}"),
                row=3, col=1
            )
            fig.update_layout(height=720, width=1080, title_text="Disturbance Test")

        return fig

        
    