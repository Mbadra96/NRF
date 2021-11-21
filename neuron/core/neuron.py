from neuron.core.params_loader import TN, REFRACTORY_PERIOD, R, C_SYN


class Neuron:
    def __init__(self, dt) -> None:
        self.v = 0
        self.refractory_state = False
        self.s = False
        self.Isyn = 0
        self.refractory_counter = 0

        # self.kernel = int(Neuron.kernel/dt)
        # self.n_max = Neuron.kernel / Neuron.RefractoryPeriod
        # self.potential_kernel = np.zeros([self.kernel,1],dtype=np.float32)
        # self.rate = 0
        # self.counter = 0

    def charge(self, Isyn):
        self.Isyn += Isyn

    def step(self, I, t, dt):

        if self.refractory_counter < 0:
            self.refractory_state = False

            # update (v) based on input current and connected synapses
        if not self.refractory_state:
            self.v += (dt / TN) * (-self.v + R * self.Isyn)  # update (v) using Euler Method

        # update spike trains
        if self.v >= 1.0:
            self.s = True
            self.refractory_state = True
            self.refractory_counter = int(REFRACTORY_PERIOD / dt) - 1
            self.v = 0
        elif self.v < 0:
            self.v = 0
            self.s = False
        else:
            self.s = False

        self.Isyn = 0
        self.refractory_counter -= 1

        return self.s, self.v


class OutputNeuron(Neuron):
    def __init__(self, dt) -> None:
        self.v = 0
        self.refractory_state = False
        self.s = False
        self.Isyn = 0
        self.refractory_counter = 0

    def charge(self, Isyn):
        self.Isyn += Isyn

    def step(self, I, t, dt):

        if self.refractory_counter < 0:
            self.refractory_state = False

            # update (v) based on input current and connected synapses
        if not self.refractory_state:
            self.v += (dt / TN) * (-self.v + R * self.Isyn)  # update (v) using Euler Method

        # update spike trains
        if self.v >= 1.0:
            self.s = True
            self.refractory_state = True
            self.refractory_counter = int(REFRACTORY_PERIOD / dt) - 1
            self.v = 0
        elif self.v < 0:
            self.v = 0
            self.s = False
        else:
            self.s = False

        self.Isyn = 0
        self.refractory_counter -= 1

        return self.s, self.v


class InputNeuron(Neuron):
    def __init__(self, dt) -> None:
        self.v = 0
        self.refractory_state = False
        self.s = False
        self.Isyn = 0
        self.refractory_counter = 0

    def step(self, I: float, t, dt):
        if self.refractory_counter < 0:
            self.refractory_state = False

            # update (v) based on input current and connected synapses
        if not self.refractory_state:
            # self.v += (dt/Neuron.Tn)*(-(self.v-Neuron.Vr) + Neuron.R*(self.Isyn)) # update (v) using Euler Method
            self.v += (dt / TN) * (-self.v + R * (C_SYN * I))

        # update spike trains
        if self.v >= 1.0:
            self.s = True
            self.refractory_state = True
            self.refractory_counter = int(REFRACTORY_PERIOD / dt) - 1
            self.v = 0
        elif self.v < 0:
            self.v = 0
            self.s = False
        else:
            self.s = False

        self.refractory_counter -= 1

        return self.s, self.v
