from neuron.optimizer.neat.genome import Genome
from pathlib import Path  # type: ignore
from neuron.core.params_loader import TIME_STEP, SAMPLES, t
import matplotlib.pyplot as plt
import time


class Timer:
    def __init__(self, name: str = ""):
        self.__start = time.perf_counter()
        self.__name = name

    def stop(self):
        del self

    def __del__(self):
        end = time.perf_counter()
        print(f"{self.__name} time duration is = {(end - self.__start)*1000} ms")


if __name__ == '__main__':
    time.perf_counter()
    genome: Genome = Genome.load(f"{Path().absolute()}/scenarios/scenario_01/scenario_01")
    cont = genome.build_phenotype(TIME_STEP)
    counter = 0

    timer = Timer("For Loop")
    for i, ts in enumerate(t):
        if cont.step([1, 0], ts, TIME_STEP)[1]:
            counter += 1
    del timer

    print(counter)

