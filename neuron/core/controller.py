from neuron.core.synapse import Synapse
from neuron.core.neuron import Neuron


class NeuroController:
    def __init__(self, connection, inputs, outputs, dt) -> None:
        
        # Create Neurons
        self.neurons = []
        for i in range(len(connection)):
            self.neurons.append(Neuron(dt))
        
        # Create Synapses
        self.synapses = []

        for i in range(len(connection)):
            for j in range(len(connection[i])):
                if connection[i][j] != 0:
                    self.synapses.append(Synapse(w=connection[i][j], pre=self.neurons[i], post=self.neurons[j]))
        self.inputs = []
        self.outputs = []
        for i in range(len(connection)):
            if i in inputs:
                self.inputs.append(1)
            else:
                self.inputs.append(0)
            
            if i in outputs:
                self.outputs.append(1)
            else:
                self.outputs.append(0)
            
    def step(self, I, t, dt):
        inputs = self.inputs.copy()
        # translate input
        c = 0
        for i in range(len(self.inputs)):
            if self.inputs[i]:
                inputs[i] = I[c]
                c += 1
            
        outputs = []
        for s in self.synapses:
            s.step(t, dt)
        
        for i,n in enumerate(self.neurons):
            if self.outputs[i]:
                outputs.append(n.step(inputs[i], t, dt))
            else:
                n.step(inputs[i], t, dt)

        return outputs
