from neuron.optimizer.neat.genome import Genome
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.core import Neat
from eval_genome import eval_func, _ball_levitation_eval_func_testing
import numpy as np


class MassUncertaintyTest:
    def __init__(self):
        self.population_size = 10
        self.generations = 50
        Randomizer.seed(0)
        neat = Neat(2, 2)
        self.population = neat.generate_population(self.population_size, eval_func)

    def start(self):
        # for i in range(self.generations):
        #     self.population.update(i+1, i == self.generations, save_best=True)

        genome: Genome = Genome.load("best")
        f = _ball_levitation_eval_func_testing(genome, show=True, mass=1.0)
        m = np.arange(0.8, 1.3, 0.05)
        for i in range(len(m)):
            _ball_levitation_eval_func_testing(genome, show=True, mass=m[i], fig=f)
        f.write_image("results/MassUncertaintyTest.png")
        f.show()


