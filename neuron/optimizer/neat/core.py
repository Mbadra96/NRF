from neuron.optimizer.neat.gene_set import GeneSet
from neuron.optimizer.neat.node_gene import NodeType
from neuron.optimizer.neat.history_marking import HistoryMarking
from neuron.optimizer.neat.genome import Genome
from neuron.optimizer.neat.population import Population
from neuron.utils.randomizer import Randomizer


class Neat:
    def __init__(self, no_of_inputs: 'int', no_of_outputs: 'int') -> None:
        self.minimal_node_genes = GeneSet()
        self.minimal_connection_genes = GeneSet()
        self.no_of_inputs = no_of_inputs
        self.no_of_outputs = no_of_outputs
        self.history_marking = HistoryMarking()
        self.init()

    def init(self):
        for i in range(self.no_of_inputs):
            self.minimal_node_genes.put(self.history_marking.add_input())
    
        for i in range(self.no_of_outputs):
            self.minimal_node_genes.put(self.history_marking.add_output())

        for node_gene1 in self.minimal_node_genes:
            for node_gene2 in self.minimal_node_genes:
                if node_gene1 == node_gene2:
                    continue
                if (node_gene1.type == NodeType.INPUT) and (node_gene2.type == NodeType.OUTPUT):
                    self.minimal_connection_genes.put(
                        self.history_marking.get_connection_gene(node_gene1.innovation_number,
                                                                 node_gene2.innovation_number))

    def get_history_marking(self):
        return self.history_marking

    def get_random_genome(self):

        g = Genome(self.minimal_node_genes.clone(),self.minimal_connection_genes.clone(),self.history_marking).mutate()

        for connection_gene in g.connection_genes:
            connection_gene.weight = Randomizer.Float(-1.0, 1.0)

        return g

    def generate_population(self, size: 'int', evaluation_function):
        p = Population(size, evaluation_function)

        for i in range(size):
            p.add_genome(self.get_random_genome())
        return p
