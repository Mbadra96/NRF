from neuron.optimizer.neat.genome import Genome
from neuron.core.coder import SFDecoder, ClampEncoder
from neuron.core.params_loader import params
from neuron.utils.units import *
from math import sin, cos, pi
import numpy as np
from plotly.subplots import make_subplots # type: ignore
import plotly.graph_objects as go # type: ignore

class BiCopter:
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