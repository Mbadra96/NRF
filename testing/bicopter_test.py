from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.core import Neat
from neuron.simulation.bicopter import BiCopter

# Set Optimization Variables
POPULATION_SIZE = 10
GENERATIONS = 50

class BiCopterTest:
    def __init__(self) -> None:
        # Set Random seed
        Randomizer.seed(0)

        # init NEAT
        self.neat = Neat(BiCopter.INPUT_NEURONS, BiCopter.OUTPUT_NEURONS)

        # Generate Population
        self.population = self.neat.generate_population(POPULATION_SIZE, BiCopter.evaluate_genome)

        # Print Start Message
        print(f"Starting Neat with population of {POPULATION_SIZE} for {GENERATIONS} generations")

    def run(self):

        for i in range(GENERATIONS):

            # update population
            self.population.update(i == GENERATIONS, save_best=True)
            # print updates
            print(f"----- Generation {i+1} -----")
            print(f"Generation {i+1} Best = {self.population.best_fitness}")
            print(f"Generation {i+1} Worst = {self.population.worst_fitness}")

        print("-------------------------------------")
        print(f"No OF Species = {self.population.get_species_size()}")

