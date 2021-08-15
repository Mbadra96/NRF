from neuron.core.synapse import Synapse
from neuron.core.neuron import Neuron

class NeuroController:
    def __init__(self,connection,dt,neuron_type:'Neuron'=Neuron,synapse_type:'Synapse'=Synapse) -> None:
        
        # Create Neurons
        self.neurons = []
        for i in range(len(connection)):
            self.neurons.append(neuron_type(dt))
        
        # Create Synapses
        self.synapses = []

        for i in range(len(connection)):
            for j in range(len(connection[i])):
                if connection[i][j] != 0:
                    self.synapses.append(synapse_type(w=connection[i][j],pre=self.neurons[i],post=self.neurons[j]))

    def step(self,I,t,dt):
        output = []
        for s in self.synapses:
            s.step(t,dt)
        
        for i,n in enumerate(self.neurons):
            output.append(n.step(I[i],t,dt))

        return output