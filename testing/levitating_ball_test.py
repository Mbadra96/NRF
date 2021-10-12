from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.core import Neat
from neuron.simulation.levitating_ball import LevitatingBall

# Set Optimization Variables
POPULATION_SIZE = 50
GENERATIONS = 50

class LevitatingBallTest:
    def __init__(self) -> None:
        # Set Random seed
        Randomizer.seed(0)

        # init NEAT
        self.neat = Neat(LevitatingBall.INPUT_NEURONS, LevitatingBall.OUTPUT_NEURONS)

        # Generate Population
        self.population = self.neat.generate_population(POPULATION_SIZE, LevitatingBall.evaluate_genome)

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

