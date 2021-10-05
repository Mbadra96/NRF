from neuron.core.synapse import Synapse
from neuron.core.neuron import Neuron, InputNeuron, OutputNeuron


class NeuroController:
    def __init__(self, connection, inputs, outputs, dt) -> None:
        
        # Create Neurons
        self.neurons = []
        for i in range(len(connection)):
            if i in inputs:
                self.neurons.append(InputNeuron(dt))
            elif i in outputs:
                self.neurons.append(OutputNeuron(dt))
            else:
                self.neurons.append(Neuron(dt))
        
        # Create Synapses
        self.synapses = []

        for i in range(len(connection)):
            for j in range(len(connection[i])):
                if connection[i][j] != 0:
                    self.synapses.append(Synapse(w=connection[i][j], pre=self.neurons[i], post=self.neurons[j]))
        self.input_indices = inputs
        self.output_indices = outputs

    def step(self, I, t, dt):

        assert len(I) == len(self.input_indices), f"Size of inputs should be {len(self.input_indices)} while given {len(I)}"
        outputs = []
        inputs = [0]*len(self.neurons)
        input_list_counter = 0
        for i, current in enumerate(self.neurons):
            if i in self.input_indices:
                inputs[i] = I[input_list_counter]
                input_list_counter += 1

        for s in self.synapses:
            s.step(t, dt)
        
        for i, n in enumerate(self.neurons):
            if i in self.output_indices:
                outputs.append(n.step(inputs[i], t, dt)[0])
            else:
                n.step(inputs[i], t, dt)

        return outputs
