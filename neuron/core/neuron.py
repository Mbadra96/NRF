# from pyvis.network import Network
from neuron.utils.units import ms
import os
import json
import numpy as np



f = open(os.getcwd()+"/params.json",)
params = json.load(f)
f.close()


class Neuron:
    Vr = params['Vr']
    R = params['R']
    Tn = params['Tn'] * ms
    RefractoryPeriod = params['RP'] * ms
    # kernel = params['Kernel'] * ms
    
    def __init__(self,dt) -> None:
        self.v = 0
        self.refractory_state = False
        self.s = False
        self.Isyn = 0
        self.refractory_counter = int(Neuron.RefractoryPeriod/dt)

        # self.kernel = int(Neuron.kernel/dt)
        # self.n_max = Neuron.kernel / Neuron.RefractoryPeriod
        # self.potential_kernel = np.zeros([self.kernel,1],dtype=np.float32)
        # self.rate = 0
        # self.counter = 0

    def charge(self,Isyn):
        self.Isyn += Isyn

    def step(self,I,t,dt):
        
        if self.refractory_counter <= 0:
            self.refractory_state = False 

        # update (v) based on input current and connected synapses
        if not self.refractory_state:
            self.v += (dt/Neuron.Tn)*(-(self.v-Neuron.Vr) + Neuron.R*(self.Isyn)) # update (v) using Euler Method
        
        
        # update spike trains
        if self.v >= 1.0:
            self.s = True
            self.refractory_state = True
            self.refractory_counter = int(Neuron.RefractoryPeriod/dt)
            self.v = 0
        else:
            self.s = False

        ## decoding output
        # self.potential_kernel[self.counter] = self.s
        # self.rate = self.potential_kernel.sum()/self.n_max
        
        # self.counter += 1
        
        # if self.counter >= self.kernel:
        #     self.counter = 0
            
        self.Isyn = 0
        self.refractory_counter -= 1

        return self.s, self.v

class OutputNeuron:    
    def __init__(self,dt) -> None:
        self.v = 0
        self.refractory_state = False
        self.s = False
        self.Isyn = 0
        self.refractory_counter = int(Neuron.RefractoryPeriod/dt)

    def charge(self,Isyn):
        self.Isyn += Isyn

    def step(self,I,t,dt):
        
        if self.refractory_counter <= 0:
            self.refractory_state = False 

        # update (v) based on input current and connected synapses
        if not self.refractory_state:
            self.v += (dt/Neuron.Tn)*(-(self.v-Neuron.Vr) + Neuron.R*(self.Isyn)) # update (v) using Euler Method
        
        
        # update spike trains
        if self.v >= 1.0:
            self.s = True
            self.refractory_state = True
            self.refractory_counter = int(Neuron.RefractoryPeriod/dt)
            self.v = 0
        else:
            self.s = False
            
        self.Isyn = 0
        self.refractory_counter -= 1

        return self.s, self.v
    
class InputNeuron:
    TS = params['Ts']
    CSYN= params['Csyn']
    def __init__(self,dt) -> None:
        self.v = 0
        self.refractory_state = False
        self.s = False
        self.Isyn = 0
        self.refractory_counter = int(Neuron.RefractoryPeriod/dt)


    def step(self,spike:bool,t,dt):
        # self.Isyn += (dt/InputNeuron.TS)*(-self.Isyn) + InputNeuron.CSYN * (1 if spike else 0)
        # if self.Isyn >= 2:
        #     self.Isyn = 2
        if self.refractory_counter <= 0:
            self.refractory_state = False 

        # update (v) based on input current and connected synapses
        if not self.refractory_state:
            # self.v += (dt/Neuron.Tn)*(-(self.v-Neuron.Vr) + Neuron.R*(self.Isyn)) # update (v) using Euler Method
            self.v += (dt/Neuron.Tn)*(-(self.v-Neuron.Vr) + Neuron.R*(InputNeuron.CSYN if spike else 0))
        
        # update spike trains
        if self.v >= 1.0:
            self.s = True
            self.refractory_state = True
            self.refractory_counter = int(Neuron.RefractoryPeriod/dt)
            self.v = 0
        else:
            self.s = False

        self.refractory_counter -= 1

        return self.s, self.v

# class NeuronPool:
#     def __init__(self,size,synapseMatrix,dt) -> None:
#         # Number of Neurons
#         self.size = size
#         # The synapses with its initial weights
#         self.synapseMatrix = synapseMatrix
#         # Init the network
#         self.init_visualization() # init visualization
#         self.view = self.net.show # set view to create html file 
        
#         self.dt = dt # set dt
#         self.t = 0 # init time
#         self.kernel = int(Neuron.kernel/self.dt)
#         self.n_max = Neuron.kernel / Neuron.RefractoryPeriod
        
#         self.v = np.zeros([self.size,1],dtype=np.float32) # init action potential
#         self.ones = np.ones([self.size,1],dtype=np.float32) # init one matrix to be compared with as threshold
#         self.refrectory_ref = np.zeros([self.size,1],dtype=np.float32)
#         self.refrectory_ref.fill(Neuron.RefractoryPeriod)
#         self.synapses = np.array(synapseMatrix,dtype=np.float32) # init weights of synapses
#         self.Isyn = np.zeros([self.size,1],dtype=np.float32) # 
#         self.refrectory_states = np.ones([self.size,1],dtype=np.float32)
        
#         self.ts = np.zeros([self.size,1],dtype=np.float32) # init spike time to be used in refractory period 
#         self.spikes = np.zeros([self.size,1],dtype=np.bool) # init spike train 
#         self.potential_kernel = np.zeros([self.kernel,self.size],dtype=np.float32) # init neurons rates
#         self.rates = np.zeros([self.size,1],dtype=np.float32) # init neurons rates
#         self.counter = 0  
#         self.output = np.zeros([self.size,1],dtype=np.float32)
              
#     def init_visualization(self):
#         self.net = Network(notebook=True,directed=True)
#         self.net.add_nodes(["N"+str(n+1) for n in range(self.size)])
#         for x in range(self.size):
#             for y in range(self.size):
#                 if(self.synapseMatrix[x][y] != 0):
#                     self.net.add_edge("N"+str(x+1),"N"+str(y+1),label= self.synapseMatrix[x][y])  

    
#     def step(self,I):
#         I = np.array(I,np.float32) + self.Isyn
        
#         ref = np.greater_equal(self.t - self.ts,self.refrectory_ref,dtype=np.float32)
#         self.refrectory_states = np.array(ref,dtype=np.float32)
        
#         # update (v) based on input current and connected synapses
#         self.v = self.refrectory_states*(self.v + (self.dt/Neuron.Tn)*(-(self.v-Neuron.Vr) + Neuron.R*I)) # update (v) using Euler Method
        
        
#         # update spike trains
#         np.greater_equal(self.v,self.ones,out=self.spikes,dtype=np.float32) # Check for spikes
        
#         # update Isyn
#         self.Isyn = self.Isyn + ((self.dt/Neuron.Ts)*(-self.Isyn) + Neuron.Csyn*self.synapses.transpose()@np.array(self.spikes,dtype=np.float32))
        
#         # update spike times
#         self.ts = self.t*np.array(self.spikes,dtype=np.float32) + self.ts * np.array(np.bitwise_not(self.spikes),dtype=np.float32)
        
#         # update refractory states
        
#         print(self.refrectory_states)
#         # exit(0)
#         # update (v) based on refractory states
#         # self.v = self.v * self.refrectory_states 
#         # decoding output
#         self.potential_kernel[self.counter] = np.array(self.spikes.reshape([self.size]),dtype=np.float32)
#         self.rates = self.potential_kernel.sum(axis=0).reshape([self.size,1])#/3683.14

#         self.output = self.output + self.rates

        
        
#         self.counter += 1
        
#         if(self.counter >= self.kernel):
#             self.counter = 0
            
#         self.t += self.dt
#         return self.rates
        

