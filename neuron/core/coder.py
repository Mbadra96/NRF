from typing import Tuple
import numpy as np


class StepEncoder:
    def __init__(self) -> None:
        pass

    def encode(self, signal:float) -> Tuple[int, int]:
        if signal > 0:
            return 1, 0
        elif signal < 0:
            return 0, 1
        else:
            return 0, 0


class LSEncoder:
    """ Linear Saturation Encoder implementation"""
    def __init__(self):
        pass

    def encode(self, signal: float) -> Tuple[float, float]:
        if signal > 0.0:
            if signal > 1.0:
                return 1.0, 0
            return signal, 0

        if signal < 0.0:
            if signal < -1.0:
                return 0, 1.0
            return 0, - signal
        return 0.0, 0.0


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

