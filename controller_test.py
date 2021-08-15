from neuron.core.controller import NeuroController
from neuron.utils.units import *
import matplotlib.pyplot as plt
from math import sin,pi
import numpy as np
TIME = 0.5 * sec
TIMESTEP = 0.5 * ms # dt is 0.1 ms
SAMPLES = int(TIME/TIMESTEP)

t = np.arange(0,TIME,TIMESTEP)
v1 = [0]*SAMPLES
v2 = [0]*SAMPLES


print(f"Simulation of Neuron Controller for TIME = {TIME} sec and TIMESTEP = {TIMESTEP} s with SAMPLES = {SAMPLES}")


if __name__ == "__main__":
    n = NeuroController([[0,2],[0,0]],TIMESTEP)

    for i in range(SAMPLES):
        output = n.step([1,0],t[i],TIMESTEP)
        v1[i] = output[0][0]
        v2[i] = output[1][0]

    plt.subplot(211)
    plt.plot(t,v1)
    plt.grid()
    
    plt.subplot(212)
    plt.plot(t,v2)
    plt.grid()
    
    plt.show()    
       