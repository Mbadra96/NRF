import math 

import matplotlib.pyplot as plt
import numpy as np
import random
class SFEncoder:
    def __init__(self,threshold) -> None:
        self.base = None
        self.threshold = threshold
        self.started = False

    def encode(self,signal:float):
        # Based on algorithm provided in:
        #   Petro et al. (2020)
        if not self.base:
            self.base = signal
            return 0, 0

        if signal > self.base + self.threshold:
            self.base = self.base + self.threshold
            return 1, 0

        elif signal < self.base - self.threshold:
            self.base = self.base - self.threshold
            return 0, 1

        return 0, 0

class SFDecoder:
    def __init__(self,base,threshold) -> None:
        self.base = base
        self.threshold = threshold

    def decode(self,spike1:int,spike2:int)->float:
        if spike1 > 0 and spike2 == 0:
            self.base = self.base + self.threshold
            
        elif spike1 == 0 and spike2 > 0:
            self.base = self.base - self.threshold
        
        return self.base
        

if __name__ == "__main__":
    encoder = SFEncoder(0.1)
    decoder = SFDecoder(0,0.1)

    t = np.arange(0,5,0.01)
    orignal = [0.0]*len(t)
    constructed = [0.0]*len(t)

    for i in range(len(t)):
        orignal[i] = math.sin(2*math.pi*t[i])
        spike1,spike2 = encoder.encode(orignal[i])
        constructed[i] = decoder.decode(spike1,spike2)

    plt.plot(t,orignal)
    plt.plot(t,constructed)
    plt.show()


        