from neuron.utils.randomizer import Randomizer
# from neuron.optimizer.neat.genome import Genome
# from neuron.optimizer.neat.gene_set import GeneSet
# from neuron.optimizer.neat.node_gene import NodeGene,NodeType
# from neuron.optimizer.neat.connection_gene import ConnectionGene
# from neuron.optimizer.neat.history_marking import HistoryMarking
# from neuron.optimizer.neat.species import Species,Member
from neuron.optimizer.neat.core import Neat
if __name__ == "__main__":
    # G = GeneSet()
    # G1 = NodeGene(6,NodeType.INPUT)
    # G2 = NodeGene(7,NodeType.HIDDEN)
    # G.put(G1)
    # G.put(G2)
    # C1 = ConnectionGene(1,G1.innovation_number,G2.innovation_number,1.0,True)

    # C2 = C1.clone()
    # C2.innovation_number = 7

    # print(C2==C1)
    # # for gene in G:
    # #     print(gene)
    # # print(G.get_random_gene())

    # H = HistoryMarking()

    # n1 = H.add_input()
    # n2 = H.add_output()

    # c1 = H.get_connection_gene(n1.innovation_number,n2.innovation_number)
    # n3 = H.get_node_gene(c1)
    # print(H)
    # Randomizer.seed(2)
    # S = Species(Member(Genome()))
    # for i in range(10):
    #     S.add(Member(Genome(),Randomizer.Float(0.0,1.0)))
    # S.sort()
    # S.elimnate()
    # print(Randomizer.choice(S.members).fitness)
    # it = G.__iter__()
    # print(next(it))
    # print(it.__length_hint__())
    # print(next(it))
    # print(it.__length_hint__())
    Randomizer.seed(10)
    n = Neat(3,3)
    g = n.get_random_genome()
    # print(g)
    n = g.build_phenotype(0.0005)
    # print(n)
    print(n.step([0.1,0.1,0.1,0,0,0,0],0.1,0.0005))