from neuron.optimizer.neat.genome import Genome
from neuron.core.coder import SFDecoder
from neuron.utils.functions import clamp_signal_to_spikes
from neuron.core.params_loader import params
from neuron.utils.units import *
from math import sin, cos, pi
import numpy as np
from plotly.subplots import make_subplots # type: ignore
import plotly.graph_objects as go # type: ignore


TIME = params["Evaluation_Time"] * sec
TIMESTEP = params["Time_Step"] * ms  # dt is 0.5 ms
SAMPLES = int(TIME / TIMESTEP)
t = np.arange(0, TIME, TIMESTEP)


class BiCopter:

    INPUT_NEURONS = 2
    OUTPUT_NEURONS = 4
    def __init__(self, m=0.87, g=9.81, row=1.225,
                 h=0.085, l=0.175, k_t=6.46,
                 i_x=0.0043, i_y=0.0142, i_z=0.0176,
                 i_xz=0.0001, x_dot_0=0, y_dot_0=0, z_dot_0=0,
                 roll_dot_0=0, pitch_dot_0=0, yaw_dot_0=0,
                 x_0=0, y_0=0, z_0=0,
                 roll_0=0, pitch_0=0, yaw_0=0) -> None:
        self.m = m
        self.g = g
        self.row = row
        self.h = h
        self.l = l
        self.k_t = k_t
        self.i_x = i_x
        self.i_y = i_y
        self.i_z = i_z
        self.i_xz = i_xz
        self.x_dot = x_dot_0
        self.y_dot = y_dot_0
        self.z_dot = z_dot_0
        self.roll_dot = roll_dot_0
        self.pitch_dot = pitch_dot_0
        self.yaw_dot = yaw_dot_0
        self.x = x_0
        self.y = y_0
        self.z = z_0
        self.roll = roll_0
        self.pitch = pitch_0
        self.yaw = yaw_0

    def step(self, w1, w2, t, dt):
        u1 = self.k_t * ((w1 ** 2) + (w2 ** 2))
        u2 = self.k_t * ((w1 ** 2) - (w2 ** 2))

        self.x_dot += dt * ((-u1 / self.m) * (
                    sin(self.roll) * sin(self.yaw) + cos(self.roll) * sin(self.pitch) * cos(self.yaw)))

        self.y_dot += dt * ((u1 / self.m) * (
                    sin(self.roll) * cos(self.yaw) - cos(self.roll) * sin(self.pitch) * sin(self.yaw)))

        self.z_dot += dt*((-u1/self.m)*cos(self.roll)*cos(self.pitch) + self.g)

        self.roll_dot += self.l*u2/self.i_x

        self.x += dt * self.x_dot
        self.y += dt * self.y_dot
        self.z += dt * self.z_dot
        self.roll += dt * self.roll_dot

        return self.x_dot, self.y_dot, self.z_dot, self.roll_dot, self.x, self.y, self.z, self.roll

    @staticmethod
    def evaluate_genome(genome: Genome) -> float:
        output_decoder_threshold = 0.1
        output_base = 0.8127610321343152
        decoder_1 = SFDecoder(output_base, output_decoder_threshold)
        decoder_2 = SFDecoder(output_base, output_decoder_threshold)
        # decoder = MWDecoder(20, output_base, output_decoder_threshold)
        cont = genome.build_phenotype(TIMESTEP)

        roll_ref = 0.0
        roll_dot_ref = 0.0
        total_error = 0.0
        roll = 1.0
        roll_dot = 0.0
        copter = BiCopter(roll_0=roll)

        # Simulation Loop
        for i in range(SAMPLES):
            e = (roll - roll_ref) + (roll_dot - roll_dot_ref) #+ Randomizer.Float(-disturbance_magnitude, disturbance_magnitude)
            total_error += abs((roll - roll_ref) / 10.0)
            ###########################
            out = cont.step(clamp_signal_to_spikes(e), t[i], TIMESTEP)
            w1 = decoder_1.decode(out[0],out[1])  # Controller
            w2 = decoder_2.decode(out[2],out[3])  # Controller
            _, _, _, roll_dot, _, _, _, roll = copter.step(w1, w2, t[i], TIMESTEP)  # Model

        return total_error

    @staticmethod
    def evaluate_genome_with_figure(genome: Genome, fig=None) -> go.Figure:
        output_decoder_threshold = 0.1
        output_base = 0.8127610321343152
        decoder_1 = SFDecoder(output_base, output_decoder_threshold)
        decoder_2 = SFDecoder(output_base, output_decoder_threshold)
        # decoder = MWDecoder(20, output_base, output_decoder_threshold)
        cont = genome.build_phenotype(TIMESTEP)

        roll_ref = 0.0
        roll_dot_ref = 0.0
        total_error = 0.0
        roll = 1.0
        roll_dot = 0.0
        copter = BiCopter(roll_0=roll)

        v1 = [0.0] * SAMPLES
        v2 = [0.0] * SAMPLES
        v3 = [0.0] * SAMPLES
        v4 = [0.0] * SAMPLES

        # Simulation Loop
        for i in range(SAMPLES):
            e = (roll - roll_ref) + (roll_dot - roll_dot_ref) #+ Randomizer.Float(-disturbance_magnitude, disturbance_magnitude)
            total_error += abs((roll - roll_ref) / 10.0)
            ###########################
            out = cont.step(clamp_signal_to_spikes(e), t[i], TIMESTEP)
            w1 = decoder_1.decode(out[0],out[1])  # Controller
            w2 = decoder_2.decode(out[2],out[3])  # Controller
            _, _, _, roll_dot, _, _, _, roll = copter.step(w1, w2, t[i], TIMESTEP)  # Model

            v1[i], v2[i] = roll, roll_dot
            v3[i] = w1
            v4[i] = w2


        if not fig:
            fig = make_subplots(rows=4, cols=1, subplot_titles=("roll", "roll Dot", "w1","w2"))

        fig.add_trace(
            go.Scatter(x=t, y=v1, name=f"roll"),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=t, y=v2, name=f"roll dot"),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=t, y=v3, name=f"w1"),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=v4, name=f"w2"),
            row=4, col=1
        )
        fig.update_layout(height=720, width=1080, title_text="Disturbance Test")

        return fig


if __name__ == '__main__':
    TIME = 10
    TIME_STEP = 0.5 * 0.001  # dt is 0.1 ms
    SAMPLES = int(TIME / TIME_STEP)
    t = np.arange(0, TIME, TIME_STEP)
    bi_copter = BiCopter()
    x = [0] * SAMPLES
    y = [0] * SAMPLES
    z = [0] * SAMPLES
    roll = [0] * SAMPLES
    w = 0.8127610321343152

    for i in range(SAMPLES):
        _, _, _, _, x[i], y[i], z[i], roll[i] = bi_copter.step(w+sin(t[i]*4*pi)*0.01, w+sin(t[i]*4*pi)*0.01, t[i], TIME_STEP)

    fig = make_subplots(rows=4, cols=1)

    fig.add_trace(
        go.Scatter(x=t, y=x, name="x"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=t, y=y, name="y"),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=t, y=z, name="z"),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=roll, name="roll"),
        row=4, col=1
    )
    fig.update_layout(height=720, width=1080, title_text="BI-COPTER TEST")
    fig.show()
