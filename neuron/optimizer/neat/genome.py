from neuron.optimizer.neat.node_gene import NodeType
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.gene_set import GeneSet
from neuron.optimizer.neat.history_marking import HistoryMarking
from neuron.core.neuro_controller import NeuroController
from multipledispatch import dispatch
import pickle
from pyvis.network import Network


class Genome:
    C1 = 1.0
    C2 = 1.0
    C3 = 0.4
    NODEMUTATIONPROBABILITY = 30
    CONNECTIONMUTATIONPROBABILITY = 10 + NODEMUTATIONPROBABILITY
    WEIGHTMUTATIONPROBABILITY = 60 + CONNECTIONMUTATIONPROBABILITY

    @dispatch()
    def __init__(self) -> None:
        self.node_genes = GeneSet()
        self.connection_genes = GeneSet()
        self.history_marking = None
        
    @dispatch(object)
    def __init__(self,other:'Genome') -> None:
        self.node_genes = other.node_genes.clone()
        self.connection_genes = other.connection_genes.clone()
        self.history_marking = other.history_marking

    @dispatch(object,object,object)
    def __init__(self,node_genes:'GeneSet',connection_genes:'GeneSet',history_marking:'HistoryMarking') -> None:
        self.node_genes = node_genes
        self.connection_genes = connection_genes
        self.history_marking = history_marking
        
        
    def mutate_node(self):
        
        selected_connection_gene = self.connection_genes.get_random_gene()

        while not selected_connection_gene.enabled:
            selected_connection_gene = self.connection_genes.get_random_gene()  

        new_node_gene = self.history_marking.get_node_gene(selected_connection_gene)

        if new_node_gene == None:
            return
        
        new_connection_gene_1 = self.history_marking.get_connection_gene(
            selected_connection_gene.get_from(),
            new_node_gene.innovation_number
        )
        new_connection_gene_2 = self.history_marking.get_connection_gene(
            new_node_gene.innovation_number,
            selected_connection_gene.get_to()
        )

        if (new_connection_gene_1 == None) or (new_connection_gene_2 == None):
            return

        new_connection_gene_1.weight = selected_connection_gene.weight
        selected_connection_gene.enabled = False
        self.node_genes.put(new_node_gene)
        self.connection_genes.put(new_connection_gene_1)
        self.connection_genes.put(new_connection_gene_2)
    
    def mutate_connection(self):
        g1 = self.node_genes.get_random_gene()
        g2 = self.node_genes.get_random_gene()

        while not g1 == g2:
            g2 = self.node_genes.get_random_gene()

        new_connection_gene = self.history_marking.get_connection_gene(g1.innovation_number,g2.innovation_number)

        if new_connection_gene:
            self.connection_genes.put(new_connection_gene)

    def mutate_weight(self):
        self.connection_genes.get_random_gene().weight = Randomizer.Float(-1.0,1.0)

    def mutate(self) ->'Genome':
        random_number = Randomizer.Integer(0,100)

        if random_number < Genome.NODEMUTATIONPROBABILITY:
            self.mutate_node()
        elif random_number < Genome.CONNECTIONMUTATIONPROBABILITY:
            self.mutate_connection()
        else:
            self.mutate_weight()

        return self

    def get_mutated_child(self) -> 'Genome':
        return self.clone().mutate()

    def crossover(self,other:'Genome') ->'Genome':
        if self == other : return self.clone()

        # Create Node Genes
        new_node_genes = GeneSet()
        found_in_g1 = False
        found_in_g2 = False 

        for node_gene in self.history_marking.get_node_genes():
            if self.node_genes.contains(node_gene.innovation_number):
                found_in_g1 = True
            if other.node_genes.contains(node_gene.innovation_number):
                found_in_g2 = True

            if  found_in_g1 and found_in_g2 :
                if Randomizer.Integer(0,100) <= 50:
                    new_node_genes.put(self.node_genes.get(node_gene.innovation_number).clone())
                else:
                    new_node_genes.put(other.node_genes.get(node_gene.innovation_number).clone())
            elif found_in_g1 and (not found_in_g2):
                new_node_genes.put(self.node_genes.get(node_gene.innovation_number).clone())      
            
            elif (not found_in_g1) and found_in_g2:
                new_node_genes.put(other.node_genes.get(node_gene.innovation_number).clone())
            else:
                pass
            found_in_g1 = False
            found_in_g2 = False

        # Create Node Genes
        new_connection_genes = GeneSet()
        found_in_g1 = False
        found_in_g2 = False 

        for connection_gene in self.history_marking.get_connection_genes():
            if self.connection_genes.contains(connection_gene.innovation_number):
                found_in_g1 = True
            if other.connection_genes.contains(connection_gene.innovation_number):
                found_in_g2 = True

            if  found_in_g1 and found_in_g2 :
                if Randomizer.Integer(0,100) <= 50:
                    new_connection_genes.put(self.connection_genes.get(connection_gene.innovation_number).clone())
                else:
                    new_connection_genes.put(other.connection_genes.get(connection_gene.innovation_number).clone())
            elif found_in_g1 and (not found_in_g2):
                new_connection_genes.put(self.connection_genes.get(connection_gene.innovation_number).clone())      
            
            elif (not found_in_g1) and found_in_g2:
                new_connection_genes.put(other.connection_genes.get(connection_gene.innovation_number).clone())
            else:
                pass
            found_in_g1 = False
            found_in_g2 = False

        return Genome(new_node_genes,new_connection_genes,self.history_marking)

   
    def get_distance(self,other:'Genome') -> 'float':
        if (self == other):
            return 0.0

        no_of_disjoint_genes = 0
        no_of_excess_genes = 0
        no_of_genes = 0

        # Node Analysis
        it1 = self.node_genes.__iter__()
        it2 = other.node_genes.__iter__()

        g1 = next(it1)
        g2 = next(it2)

        no_of_genes += 1
        # Node disjoint genes
        while (it1.__length_hint__() > 0) and (it2.__length_hint__() > 0):
            no_of_genes += 1
            if g1 == g2:
                g1 = next(it1)
                g2 = next(it2)
                continue
            elif g1.innovation_number > g2.innovation_number:
                g2 = next(it2)
            else:
                g1 = next(it1)

            no_of_disjoint_genes += 1
        # Node excess genes
        while (it1.__length_hint__() > 0) or (it2.__length_hint__() > 0):
            no_of_genes += 1
            no_of_excess_genes += 1

            if it1.__length_hint__() > 0:
                g1 = next(it1)
            else:
                g2 = next(it2)

        # Connection Analysis
        cit1 = self.connection_genes.__iter__()
        cit2 = other.connection_genes.__iter__()

        cg1 = next(cit1)
        cg2 = next(cit2)

        weight_sum_1 = cg1.weight if cg1.enabled else 0.0
        weight_sum_2 = cg2.weight if cg2.enabled else 0.0
        no_of_genes += 1
        # Connection disjoint genes
        while (cit1.__length_hint__() > 0) and (cit2.__length_hint__() > 0):
            no_of_genes += 1
            if cg1 == cg2:
                cg1 = next(cit1)
                cg2 = next(cit2)
                weight_sum_1 += cg1.weight if cg1.enabled else 0.0
                weight_sum_2 += cg2.weight if cg2.enabled else 0.0
                continue

            elif cg1.innovation_number > cg2.innovation_number:
                cg2 = next(cit2)
                weight_sum_2 += cg2.weight if cg2.enabled else 0.0
            else:
                cg1 = next(cit1)
                weight_sum_1 += cg1.weight if cg1.enabled else 0.0

            no_of_disjoint_genes += 1
        # Node excess genes
        while (cit1.__length_hint__() > 0) or (cit2.__length_hint__() > 0):
            no_of_genes += 1
            no_of_excess_genes += 1

            if cit1.__length_hint__() > 0:
                cg1 = next(cit1)
                weight_sum_1 += cg1.weight if cg1.enabled else 0.0
            else:
                cg2 = next(cit2)
                weight_sum_2 += cg2.weight if cg2.enabled else 0.0

        abs_sum_of_weights = abs(weight_sum_1 - weight_sum_2)

        return (Genome.C1*no_of_disjoint_genes/no_of_genes) + (Genome.C2*no_of_excess_genes/no_of_genes) + Genome.C3*abs_sum_of_weights
        

    def clone(self) ->'Genome':
        return Genome(self)

    def __str__(self) -> str:
                return "Genome {\n" + str(self.node_genes) + "\n" + str(self.connection_genes) + "\n\t}" 

    def build_phenotype(self, time_step) -> 'NeuroController':
        n = self.history_marking.node_genes_counter
        inputs = []
        outputs = []
        for i,gene in enumerate(self.history_marking.node_genes):
            if gene.type == NodeType.INPUT:
                inputs.append(i)
            elif gene.type == NodeType.OUTPUT:
                outputs.append(i)

        connection_matrix = [[]]
        for i in range(n):
            for j in range(n):
                connection_matrix[i].append(0.0)
            if i < n-1:
                connection_matrix.append([])

        for connection_gene in self.connection_genes:
            if connection_gene.enabled:
                connection_matrix[connection_gene.f][connection_gene.t] = connection_gene.weight
                
        return NeuroController(connection_matrix,inputs,outputs,time_step)

    def visualize(self, name:str='genome'):
        self.net = Network(directed=True)

        for node in self.node_genes:
            if node.type == NodeType.INPUT:
                self.net.add_node(node.innovation_number,label=f"N{node.innovation_number}",color="blue")

            elif node.type == NodeType.OUTPUT:
                self.net.add_node(node.innovation_number,label=f"N{node.innovation_number}",color="red")
            else:
                self.net.add_node(node.innovation_number,label=f"N{node.innovation_number}",color="green")
        
        for conn in self.connection_genes:
            if conn.enabled:
                self.net.add_edge(conn.f,conn.t,title=f"C{conn.innovation_number} : {conn.weight}")

        self.net.show(name+".html")


    def save(self,name:str="untitled"):
        f = open(f'{name}.genome', 'wb')
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load(name:str="untitled"):
        f = open(f'{name}.genome', 'rb')
        g = pickle.load(f)
        f.close()
        return g