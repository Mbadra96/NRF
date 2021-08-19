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

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def eval_func(genome,show:bool=False)->float: 
    cont = genome.build_phenotype(TIMESTEP)

    if show:
        v1 = [0]*SAMPLES
        v2 = [0]*SAMPLES
        v3 = [0]*SAMPLES
    K = 20.0


    ball = LevitatingBall(1,0,0)
    x_ref = 5
    x_dot_ref = 0
    total_error = 0
    total_F = 0
    F = 0
    x = 0
    x_dot = 0

    # Simulation Loop
    for i in range(SAMPLES):
        e = (x_ref - x) + (x_dot_ref - x_dot)
        total_error += abs((x_ref - x)/10.0) 
        ###########################
        sensors = [*clamp(e)]
        ######################
        output = cont.step(sensors,t[i],TIMESTEP) # Controller 
        x, x_dot = ball.step(F,t[i],TIMESTEP) # Model
        
        
        if show:
            v1[i],v2[i]= x, x_dot
            v3[i] = F

        ######################
        F = K*(output[0][0] - output[1][0])
        total_F += abs(F/K)
        ######################
    
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
        fig.update_layout(height=720, width=1080, title_text="Genome Test")
        fig.show()
    
    return total_error#+0.01*total_F

if __name__ == "__main__":
    genome:Genome = Genome.load("best")
    print(genome)
    genome.visualize()
    print(eval_func(genome,show=True))
