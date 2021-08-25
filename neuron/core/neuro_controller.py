from neuron.core.synapse import Synapse
from neuron.core.neuron import Neuron,InputNeuron,OutputNeuron
from neuron.core.coder import SFDecoder, SFEncoder

class NeuroController:
    def __init__(self,input_signals,output_signals,input_encoder_threshold,output_decoder_threshold,output_base,connection,inputs,outputs,dt) -> None:
        
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
                    self.synapses.append(Synapse(w=connection[i][j],pre=self.neurons[i],post=self.neurons[j]))
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
            
        self.encoders = []
        self.decoders = []

        for _ in range(input_signals):
            self.encoders.append(SFEncoder(input_encoder_threshold))

        for _ in range(output_signals):
            self.decoders.append(SFDecoder(base=output_base,threshold=output_decoder_threshold))
    
    def step(self,I,t,dt):
        
        # if(len(I) != len(self.encoders)):
        #     raise IndexError(f"input length: {len(I)} not equal encoders length: {len(self.encoders)}")
        outputs = []
        inputs = [0]*len(self.neurons)
        inputs[0],inputs[1] = I[0],I[1]

        # for i, encoder in enumerate(self.encoders):
        #     inputs[i*2],inputs[i*2 + 1] = encoder.encode(I[i])
            


        for s in self.synapses:
            s.step(t,dt)
        
        for i,n in enumerate(self.neurons):
            if self.outputs[i]:
                outputs.append(n.step(inputs[i],t,dt))
            else:
                n.step(inputs[i],t,dt)

        output_signals = []

        for i, decoder in enumerate(self.decoders):
            output_signals.append(decoder.decode(int(outputs[i*2][0]),int(outputs[i*2 + 1][0])))

        return output_signals
        # return int(outputs[0][0]),int(outputs[1][0])