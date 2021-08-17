from neuron.core.controller import NeuroController
from neuron.simulation.levitating_ball import LevitatingBall
from neuron.utils.randomizer import Randomizer
from neuron.utils.units import *
import numpy as np
# from neuron.optimizer.neat.genome import Genome
# from neuron.optimizer.neat.gene_set import GeneSet
# from neuron.optimizer.neat.node_gene import NodeGene,NodeType
# from neuron.optimizer.neat.connection_gene import ConnectionGene
# from neuron.optimizer.neat.history_marking import HistoryMarking
# from neuron.optimizer.neat.species import Species,Member
from neuron.optimizer.neat.core import Neat
TIME = 10 * sec
TIMESTEP = 0.5 * ms # dt is 0.1 ms
SAMPLES = int(TIME/TIMESTEP)

def clamp(x):
    if x > 1:
        return 1, 0
    elif x < 0 :
        if x < -1:
            return 0, 1
        else:
            return 0, -x
    else:
        return x , 0

def eval_func(genome)->float: 
    cont = genome.build_phenotype(TIMESTEP)
    t = np.arange(0,TIME,TIMESTEP)
    K = 20
    ball = LevitatingBall(1,0,0)
    x_ref = 8
    x_dot_ref = 0
    total_error = 0
    F = 0
    for i in range(SAMPLES):
        x, x_dot = ball.step(F,t[i],TIMESTEP)
        e = (x_ref - x) + (x_dot_ref - x_dot)
        total_error += abs(e) 
        output = cont.step(clamp(e),t[i],TIMESTEP)
        F = 20*(output[0][0] - output[1][0])

    return total_error

if __name__ == "__main__":
    POPULATION_SIZE = 10
    GENERATIONS = 32
    Randomizer.seed(0)
    neat = Neat(2, 2)
    population = neat.generate_population(POPULATION_SIZE,eval_func)

    for i in range(GENERATIONS):
        population.update(i)

    population.stop()

    g = population.get(0).fitness

    print(f"Best Error = {population.get(0).fitness}")
    print(f"No OF Species = {population.get_species_size()}")