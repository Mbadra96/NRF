from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.core import Neat
from eval_genome import eval_func

if __name__ == "__main__":
    POPULATION_SIZE = 10
    GENERATIONS = 50
    Randomizer.seed(0)
    neat = Neat(2, 2)
    population = neat.generate_population(POPULATION_SIZE, eval_func)
    print(f"Starting Neat with population of {POPULATION_SIZE} for {GENERATIONS} generations")

    for i in range(GENERATIONS):
        population.update(i+1, i == GENERATIONS,save_best= True)

    print(f"No OF Species = {population.get_species_size()}")
