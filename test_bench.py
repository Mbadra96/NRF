from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.core import Neat
from and_gate_eval_genome import eval_func

if __name__ == "__main__":
    POPULATION_SIZE = 100
    GENERATIONS = 20
    Randomizer.seed(0)
    neat = Neat(2, 1)
    population = neat.generate_population(POPULATION_SIZE,eval_func)

    print(f"Starting Neat with poulation of {POPULATION_SIZE} for {GENERATIONS} generations")

    for i in range(GENERATIONS):
        population.update(i+1,i==GENERATIONS,save_best= True)

    print(f"Best Error = {population.get(0).fitness}")
    print(f"No OF Species = {population.get_species_size()}")