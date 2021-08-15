from neuron.utils.units import *
from neuron.core.controller import NeuroController
from neuron.simulation.levitating_ball import LevitatingBall
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from math import sin,pi

TIME = 50 * sec
TIMESTEP = 0.5 * ms # dt is 0.1 ms
SAMPLES = int(TIME/TIMESTEP)

t = np.arange(0,TIME,TIMESTEP)
v1 = [0]*SAMPLES
v2 = [0]*SAMPLES
F = [0]*SAMPLES

print(f"Simulation of Neuron Pool for TIME = {TIME} sec and TIMESTEP = {TIMESTEP} ms with SAMPLES = {SAMPLES}")


if __name__ == "__main__":
    x_ref = 5
    x_dot_ref = 0
    ball = LevitatingBall(1,4,0)
    controller = NeuroController([[0,0.5],[0,0]],TIMESTEP)

    v1[0],v2[0]=ball.step(0,t[0],TIMESTEP)

    e = (x_ref - v1[0]) + (x_dot_ref - v2[0])

    if e > 1 :
        e = 1
    elif e < 0 :
        e = 0

    out=controller.step([e,0],t[0],TIMESTEP)
    F[0] = out[1][0]
    for i in range(1,SAMPLES,1):   
        
        v1[i],v2[i]=ball.step(20*out[1][0],t[i],TIMESTEP)
        e = (x_ref - v1[i]) #+ (x_dot_ref - v2[i])

        if e > 1 :
            e = 1
        elif e < 0 :
            e = 0

        out=controller.step([e,0],t[0],TIMESTEP)
        F[i] = out[1][0]

    fig = make_subplots(rows=3, cols=1,subplot_titles=("X", "X DOT", "F"))

    fig.add_trace(
        go.Scatter(x=t, y=v1),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=t, y=v2),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=t, y=F),
        row=3, col=1
    )
    # fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    fig.show()
    