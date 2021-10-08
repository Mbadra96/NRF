from neuron.optimizer.neat.gene_set import GeneSet
from neuron.optimizer.neat.node_gene import NodeGene, NodeType
from neuron.optimizer.neat.connection_gene import ConnectionGene
from typing import Optional

class HistoryMarking:
    def __init__(self) -> None:
        self.node_genes = GeneSet()
        self.connection_genes = GeneSet()

        self.node_genes_counter = 0
        self.connection_genes_counter = 0

    def get_connection_genes(self) -> 'GeneSet':
        return self.connection_genes
    
    def get_node_genes(self) -> 'GeneSet':
        return self.node_genes

    def create_new_node(self,node_type:'NodeType') ->'NodeGene':
        self.node_genes_counter += 1
        node = self.node_genes.put(NodeGene(self.node_genes_counter-1,node_type))
        return node.clone() # type: ignore

    def create_new_connection(self,input_innovation_number,output_innovation_number):
        self.connection_genes_counter += 1
        connection =  self.connection_genes.put(ConnectionGene(self.connection_genes_counter-1,input_innovation_number,output_innovation_number,1.0,True))
        new_conn =  connection.clone()
        # new_conn.weight = Randomizer.Float(-1.0,1.0)
        return new_conn

    def add_input(self) -> 'NodeGene':
        return self.create_new_node(NodeType.INPUT)
    

    def add_output(self) -> 'NodeGene':
        return self.create_new_node(NodeType.OUTPUT)
    
    def get_connection_gene(self, input_innovation_number:int, output_innovation_number:int) -> Optional[ConnectionGene]:
        # Return NONE if input = output
        if input_innovation_number == output_innovation_number:
            return None
        
        # Search for it in the history marking
        for connection_gene in self.connection_genes:
            if (connection_gene.get_from() == input_innovation_number) and connection_gene.get_to() == output_innovation_number:
                return connection_gene.clone()
        
        # If not found add; create a new one
        return self.create_new_connection(input_innovation_number,output_innovation_number)

    def get_node_gene(self,connection_gene:'ConnectionGene')->'NodeGene':
        output_innovation_number = connection_gene.get_to()
        input_innovation_number = connection_gene.get_from()

        chosen_nodes_genes = GeneSet()
        # Try to find it
        for gene in self.connection_genes:
            if not gene == connection_gene:
                chosen_nodes_genes.put(self.node_genes.get(gene.get_to()))

        for node in chosen_nodes_genes:
            for connection in self.connection_genes:
                if (connection.get_from() == node.innovation_number) and (connection.get_to() == output_innovation_number):
                    return node.clone()

        # Create new one
        node =  self.create_new_node(NodeType.HIDDEN)
        self.create_new_connection(input_innovation_number,node.innovation_number)
        self.create_new_connection(node.innovation_number,output_innovation_number)
        return node

    def __str__(self) -> str:
        return "HistoryMarking{\n" + str(self.node_genes) + "\n" + str(self.connection_genes) + "\n\t}" 