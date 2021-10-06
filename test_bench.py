from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.core import Neat
from neuron.simulation.levitating_ball import LevitatingBall

# Set Optimization Variables
POPULATION_SIZE = 10
GENERATIONS = 50


if __name__ == "__main__":
    # Set Random seed
    Randomizer.seed(0)

    # init NEAT
    neat = Neat(LevitatingBall.INPUT_NEURONS, LevitatingBall.OUTPUT_NEURONS)

    # Generate Population
    population = neat.generate_population(POPULATION_SIZE, LevitatingBall.evaluate_genome)

    # Print Start Message
    print(f"Starting Neat with population of {POPULATION_SIZE} for {GENERATIONS} generations")

    for i in range(GENERATIONS):

        # update population
        population.update(i == GENERATIONS, save_best=True)

        # print updates
        print(f"----- Generation {i+1} -----")
        print(f"Generation {i+1} Best = {population.best_fitness}")
        print(f"Generation {i+1} Worst = {population.worst_fitness}")

    print("-------------------------------------")
    print(f"No OF Species = {population.get_species_size()}")

