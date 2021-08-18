from neuron.utils.randomizer import Randomizer
from neuron.utils.units import *
from neuron.optimizer.neat.core import Neat
from eval_genome import eval_func

if __name__ == "__main__":
    POPULATION_SIZE = 100
    GENERATIONS = 100
    Randomizer.seed(0)
    neat = Neat(3, 2)
    population = neat.generate_population(POPULATION_SIZE,eval_func)
    print(f"Starting Neat with poulation of {POPULATION_SIZE} for {GENERATIONS} generations")
    for i in range(GENERATIONS):
        population.update(i,i==(GENERATIONS-1))

        population.get(0).genome.save("best")

    print(f"Best Error = {population.get(0).fitness}")
    print(f"No OF Species = {population.get_species_size()}")