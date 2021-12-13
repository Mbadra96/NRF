from neuron.optimizer.neat.genome import Genome
from pathlib import Path  # type: ignore
from neuron.core.params_loader import TIME_STEP, SAMPLES, t

from neuron.utils.timer import Timer


if __name__ == '__main__':
    genome: Genome = Genome.load(f"{Path().absolute()}/scenarios/scenario_01/scenario_01")
    cont = genome.build_phenotype(TIME_STEP)
    counter = 0

    timer = Timer("For Loop")
    for i, ts in enumerate(t):
        if cont.step([1, 0], ts, TIME_STEP)[1]:
            counter += 1
    timer.stop()

    print(counter)

