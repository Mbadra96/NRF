
from neuron.optimizer.neat.genome import Genome
from neuron.simulation.bicopter import BiCopter

if __name__ == "__main__":
    genome: Genome = Genome.load("best")
    BiCopter.evaluate_genome_with_figure(genome).show()