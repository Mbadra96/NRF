from neuron.simulation.levitating_ball import LevitatingBall
from neuron.utils.units import *
import numpy as np
from neuron.optimizer.neat.genome import Genome
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

TIME = 0.2 * sec
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

    ref1 = 0
    ref2 = 0.5
    ref3 = 0.5
    ref4 = 1

    total_error = 0
    output1 = []
    output2 = []
    output3 = []
    output4 = []

    # Simulation Loop
    for i in range(SAMPLES):
        output1 = cont.step([0.5,0.5],t[i],TIMESTEP) # Controller 

    for i in range(SAMPLES):
        output2 = cont.step([0.5,1],t[i],TIMESTEP) # Controller 

    for i in range(SAMPLES):
        output3 = cont.step([1,0.5],t[i],TIMESTEP) # Controller 

    for i in range(SAMPLES):
        output4 = cont.step([1,1],t[i],TIMESTEP) # Controller 
        
    if show:
        print(f"o1 : {output1[0][0]}")
        print(f"o2 : {output2[0][0]}")
        print(f"o3 : {output3[0][0]}")
        print(f"o4 : {output4[0][0]}")
    total_error = np.sqrt((ref1 - output1[0][0])**2+(ref2 - output2[0][0])**2 + (ref3 - output3[0][0])**2 + (ref4 - output4[0][0])**2)

    
    return total_error#+0.01*total_F

if __name__ == "__main__":
    genome:Genome = Genome.load("best")
    # print(genome)
    # genome.visualize()
    print(eval_func(genome,show=True))
