from neuron.simulation.levitating_ball import LevitatingBall
from neuron.utils.units import *
import numpy as np
from neuron.optimizer.neat.genome import Genome
from plotly.subplots import make_subplots
import plotly.graph_objects as go

TIME = 5 * sec
TIMESTEP = 0.5 * ms # dt is 0.1 ms
SAMPLES = int(TIME/TIMESTEP)
t = np.arange(0,TIME,TIMESTEP)

def clamp(x):
    if x > 1:
        return 1, 0
    elif x < 0 :
        if x < -1:
            return 0, 1
        else:
            return 0, -x
    else:
        return x , 0

def eval_func(genome,show:bool=False)->float: 
    cont = genome.build_phenotype(TIMESTEP)
    if show:
        v1 = [0]*SAMPLES
        v2 = [0]*SAMPLES
        v3 = [0]*SAMPLES
    K = 10
    ball = LevitatingBall(1,0,0)
    x_ref = 8
    x_dot_ref = 0
    total_error = 0
    F = 0
    for i in range(SAMPLES):
        x, x_dot = ball.step(F,t[i],TIMESTEP)
        e = (x_ref - x) + (x_dot_ref - x_dot)
        total_error += abs(e) 
        if show:
            v1[i],v2[i]=ball.step(F,t[i],TIMESTEP)
            v3[i] = F

        output = cont.step([*clamp(e),genome.bias],t[i],TIMESTEP)
        F = K*(output[0][0] - output[1][0]) + 9.81
    
    if show:
        fig = make_subplots(rows=3, cols=1, subplot_titles=("x", "x_dot", "Force"))

        fig.add_trace(
            go.Scatter(x=t, y=v1),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=t, y=v2),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=t, y=v3),
            row=3, col=1
        )
        fig.update_layout(height=720, width=1080, title_text="Side By Side Subplots")
        fig.show()
    
    return total_error

if __name__ == "__main__":
    genome = Genome.load("best")
    # print(genome)
    print(eval_func(genome,show=True))