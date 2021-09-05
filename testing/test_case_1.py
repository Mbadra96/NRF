from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.core import Neat
from eval_genome import eval_func


class TestCase1:
    def __init__(self):
        self.population_size = 10
        self.generations = 50
        Randomizer.seed(0)
        neat = Neat(2, 2)
        self.population = neat.generate_population(self.population_size, eval_func)

    def start(self):
        print(f"Starting Neat with population of {self.population_size} for {self.generations} generations")

        for i in range(self.generations):
            self.population.update(i+1, i == self.generations, save_best=True)

        print(f"No OF Species = {self.population.get_species_size()}")
