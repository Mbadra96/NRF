from neuron.utils.units import ms
from neuron.core.params_loader import params


class Neuron:
    Vr = params['Vr']
    R = params['R']
    Tn = params['Tn'] * ms
    RefractoryPeriod = params['RP'] * ms
    
    def __init__(self, dt) -> None:
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

    def charge(self, Isyn):
        self.Isyn += Isyn

    def step(self, I, t, dt):
        
        if self.refractory_counter <= 0:
            self.refractory_state = False 

        # update (v) based on input current and connected synapses
        if not self.refractory_state:
            self.v += (dt/Neuron.Tn)*(-(self.v-Neuron.Vr) + Neuron.R*self.Isyn)  # update (v) using Euler Method

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


class OutputNeuron(Neuron):    
    def __init__(self,dt) -> None:
        self.v = 0
        self.refractory_state = False
        self.s = False
        self.Isyn = 0
        self.refractory_counter = int(Neuron.RefractoryPeriod/dt)

    def charge(self, Isyn):
        self.Isyn += Isyn

    def step(self, I, t, dt):
        
        if self.refractory_counter <= 0:
            self.refractory_state = False 

        # update (v) based on input current and connected synapses
        if not self.refractory_state:
            self.v += (dt/Neuron.Tn)*(-(self.v-Neuron.Vr) + Neuron.R*(self.Isyn))  # update (v) using Euler Method

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


class InputNeuron(Neuron):
    TS = params['Ts']
    CSYN = params['Csyn']

    def __init__(self, dt) -> None:
        self.v = 0
        self.refractory_state = False
        self.s = False
        self.Isyn = 0
        self.refractory_counter = int(Neuron.RefractoryPeriod/dt)

    def step(self, spike:bool, t, dt):
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


