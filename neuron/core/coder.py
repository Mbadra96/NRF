from typing import Protocol, Tuple
import numpy as np

# Defining ENCODERS
class Encoder(Protocol):
    def encode(self, signal:float) -> Tuple[int, int]:
        """ takes a signal and return a two encoded int spikes """
class Decoder(Protocol):
    def decode(self, spike_1: int, spike_2: int) -> float:
        """ takes two spikes and returns a float decoded signal"""
class ClampEncoder:
    def __init__(self) -> None:
        pass

    def encode(self, signal:float) -> Tuple[int, int]:
        if signal > 0:
            return 1, 0
        elif signal < 0:
            return 0, 1
        else:
            return 0, 0
class SFEncoder:
    def __init__(self, threshold) -> None:
        self.base = None
        self.threshold = threshold

    def encode(self, signal: float) -> Tuple[int, int]:
        # Based on algorithm provided in:
        #   Petro et al. (2020)
        if not self.base:
            self.base = signal # type: ignore
            return 0, 0

        if signal > self.base + self.threshold:
            self.base = self.base + self.threshold
            return 1, 0

        elif signal < self.base - self.threshold:
            self.base = self.base - self.threshold
            return 0, 1

        return 0, 0
class MWEncoder:
    def __init__(self, window: int, threshold):
        self.base = None
        self.threshold = threshold
        self.window_size = window
        self.window = None
        self.counter = 0

    def encode(self, signal: float) -> Tuple[int, int]:
        # Based on algorithm provided in:
        #   Petro et al. (2020)
        if not self.base:
            self.base = signal # type: ignore
            self.window = np.full([self.window_size], self.base, dtype=np.float32) # type: ignore
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


# Defining DECODERS
class MWDecoder:
    def __init__(self, window_size, base, threshold):
        self.base = base
        self.threshold = threshold
        self.window_size = window_size
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

# class THSAEncoder:
#     def __init__(self) -> None:
#         pass
#     def encode(self, data, fir , threshold):
#         # Based on algorithm provided in:
#         #   Schrauwen et al. (2003)
#         spikes = np.zeros(len(data))
#         shift = min(data)
#         data = data - shift*np.ones(len(data))
#         for i in range(len(data)):
#             error = 0
#             for j in range(len(fir)):
#                 if i+j < len(data):
#                     if data[i+j] < fir[j]:
#                         error = error + fir[j] - data[i+j]
#             if error <= threshold:
#                 spikes[i] = 1
#                 for j in range(len(fir)):
#                     if i+j < len(data):
#                         data[i+j] = data[i+j] - fir[j]
#         return spikes, shift

# class BenDecoder:
#     def __init__(self) -> None:
#         pass

#     def decode(self, spikes, fir, shift):
#         # Based on algorithm provided in:
#         #   Petro et al. (2020)
#         #   Sengupta et al. (2017)
#         #   Schrauwen et al. (2003)
#         signal = np.convolve(spikes, fir)
#         signal = signal + shift*np.ones(len(signal))
#         signal = signal[0:(len(signal)-len(fir)+1)]
#         return signal


# import numpy as np
# from scipy import signal # type: ignore
# from scipy.stats import norm # type: ignore

# import math
# import matplotlib.pyplot as plt # type: ignore


# if __name__ == "__main__":
#     encoder = THSAEncoder()
#     decoder = BenDecoder()

#     t = np.arange(0, 5, 0.01)
#     original = [0.0] * len(t)
#     constructed = [0.0]*len(t)

#     hsa_window = [12, 15, 12]
#     hsa_fir = list()
#     hsa_fir.append(signal.triang(hsa_window[0]))
#     hsa_fir.append(norm.pdf(np.linspace(1, hsa_window[1], hsa_window[1]), 0, 5))
#     hsa_fir.append(signal.triang(hsa_window[2]))

#     hsa_m_thresholds = [0.85, 0.05, 0.5]

#     bsa_window = [9, 10, 8]
#     bsa_fir = list()
#     bsa_fir.append(signal.triang(bsa_window[0]))
#     bsa_fir.append(norm.pdf(np.linspace(1, bsa_window[1], bsa_window[1]), 1.5, 3.5))
#     bsa_fir.append(signal.triang(bsa_window[2]))

#     for i in range(len(t)):
#         original[i] = (math.sin(2 * math.pi * t[i] - math.pi/2) )
#         # spike1, spike2 = encoder.encode(original[i]+random.random()*0.01)
#         # constructed[i] = decoder.decode(spike1, spike2)

#     hsa_fir = signal.triang(hsa_window[0])


#     spikes, shift = encoder.encode(original, hsa_fir, threshold=0.1)
#     constructed = decoder.decode(spikes, hsa_fir,shift)
#     plt.plot(t, original)
#     plt.plot(t, constructed)
#     plt.show()
