from neuron.core.neuro_controller import NeuroController
from neuron.simulation.levitating_ball import LevitatingBall
from neuron.utils.units import *
import numpy as np
from neuron.optimizer.neat.genome import Genome
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import pi,sin

TIME = 50 * sec
TIMESTEP = 0.5 * ms # dt is 0.1 ms
SAMPLES = int(TIME/TIMESTEP)
t = np.arange(0,TIME,TIMESTEP)

def clamp(e):
    if e > 0:
        return 1,0
    elif e < 0:
        return 0,1
    else:
        return 0,0
    


def eval_func(genome,show:bool=False)->float: 
    input_signals = 1
    output_signals = 1
    input_encoder_threshold = 0.00001
    output_decoder_threshold = 1
    output_base = 9.81

    cont = genome.build_phenotype(input_signals,output_signals,input_encoder_threshold,output_decoder_threshold,output_base,TIMESTEP)
    # connection_matix = [[0.0,0.0,1.0,0.0],
    #                     [0.0,0.0,0.0,1.0],
    #                     [0.0,0.0,0.0,0.0],
    #                     [0.0,0.0,0.0,0.0]]
    # cont = NeuroController(input_signals,output_signals,input_encoder_threshold,output_decoder_threshold,output_base,connection_matix,[0,1],[2,3],TIMESTEP)
    
    if show:
        v1 = [0]*SAMPLES
        v2 = [0]*SAMPLES
        v3 = [0]*SAMPLES
        v4 = [0]*SAMPLES


    
    # x_ref = 2
    x_dot_ref = 0
    total_error = 0
    F = 0
    x = 1
    x_dot = 0
    e_I = 0
    ball = LevitatingBall(1,x,x_dot)

    # Simulation Loop
    for i in range(SAMPLES):
        x_ref = sin(pi*t[i]/10)+1
        e = (x_ref - x) + (x_dot_ref - x_dot)
        e_I += 0.00001*e
        total_error += abs((x_ref - x)/10.0)
        ###########################
        # sensors = [*clamp(e)]
        sensors = [e_I]
        ######################
        output = cont.step(sensors,t[i],TIMESTEP) # Controller 
        # F = 2*output[0] - 2*output[1] + 9.81 
        F = output[0]
        x, x_dot = ball.step(F,t[i],TIMESTEP) # Model
        # print(f"in -> {sensors}, out -> {output}")
        
        if show:
            v1[i],v2[i]= x, x_dot
            v3[i] = x_ref
            v4[i] = e_I 
        else:
            pass
    
    if show:
        fig = make_subplots(rows=4, cols=1, subplot_titles=("x", "x_dot", "Force","e_I"))

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
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=v4),
            row=4, col=1
        )
        fig.update_layout(height=720, width=1080, title_text="Genome Test")
        fig.show()
    
    return total_error#+0.01*total_F

if __name__ == "__main__":
    genome:Genome = Genome.load("best")
    # print(genome)
    # genome.visualize()
    print(eval_func(genome,show=True))
