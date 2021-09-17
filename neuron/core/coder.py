import math 

import matplotlib.pyplot as plt
import numpy as np
import random


class SFEncoder:
    def __init__(self, threshold) -> None:
        self.base = None
        self.threshold = threshold

    def encode(self, signal: float):
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
    def __init__(self, base, threshold) -> None:
        self.base = base
        self.threshold = threshold

    def decode(self, spike_1: int, spike_2: int) -> float:
        if spike_1 > 0 and spike_2 == 0:
            self.base = self.base + self.threshold
            
        elif spike_1 == 0 and spike_2 > 0:
            self.base = self.base - self.threshold
        
        return self.base
        

class MWEncoder:
    def __init__(self, window: int, threshold):
        self.base = None
        self.threshold = threshold
        self.window_size = window
        self.window = None
        self.counter = 0

    def encode(self, signal: float):
        # Based on algorithm provided in:
        #   Petro et al. (2020)
        if not self.base:
            self.base = signal
            self.window = np.full([self.window_size], self.base, dtype=np.float32)
            return 0, 0

        if self.counter >= self.window_size:
            self.counter = 0

        self.window[self.counter] = signal
        self.base = np.mean(self.window)

        self.counter += 1

        if signal > self.base + self.threshold:
            return 1, 0
        elif signal < self.base - self.threshold:
            return 0, 1

        return 0, 0


class MWDecoder:
    def __init__(self, window, base, threshold):
        self.base = base
        self.threshold = threshold
        self.window_size = window
        self.window = np.full([self.window_size], self.base, dtype=np.float32)
        self.counter = 0

    def decode(self, spike_1: int, spike_2: int) -> float:
        if self.counter >= self.window_size:
            self.counter = 0

        if spike_1-spike_2 > 0:
            self.base += self.threshold
        elif spike_1-spike_2 < 0:
            self.base -= self.threshold

        self.window[self.counter] = self.base

        self.counter += 1
        return float(np.mean(self.window))


if __name__ == "__main__":
    encoder = MWEncoder(10, 0.01)
    decoder = MWDecoder(10, 0, 0.01)

    t = np.arange(0, 5, 0.01)
    original = [0.0] * len(t)
    constructed = [0.0]*len(t)

    for i in range(len(t)):
        original[i] = math.sin(2 * math.pi * t[i])
        spike1, spike2 = encoder.encode(original[i]+random.random()*0.01)
        constructed[i] = decoder.decode(spike1, spike2)

    plt.plot(t, original)
    plt.plot(t, constructed)
    plt.show()


        