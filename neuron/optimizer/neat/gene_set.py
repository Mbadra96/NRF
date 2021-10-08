from typing import Dict, Optional, overload
from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.gene import Gene

class GeneSet:

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, other:'GeneSet') -> None:
       ...
        

    def __init__(self, *args, **kwargs) -> None:
        self.map:Dict[int,Gene] = dict()
        if len(args) != 0:
            other = args[0]
            if isinstance(other, GeneSet):
                for gene in other.map.values():
                    self.put(gene.clone())
            else:
                assert False, f"invaled object excepted Gene Type but {other.__class__.__name__} was given instead"
    def put(self, gene:'Gene') -> Optional[Gene]:
        if(gene == None): return None
        
        if not self.contains(gene.innovation_number):
            self.map[gene.innovation_number] = gene
        
        return gene

    def contains(self,innovation_number:int) -> bool:
        return True if innovation_number in self.map else False

    def get(self, innovation_number):
        return self.map[innovation_number]

    def __iter__(self):
        return iter(self.map.values())

    def clone(self):
        return GeneSet(self)

    def get_random_gene(self):
        return Randomizer.choice(list(self.map.values()))

    def __str__(self) -> str:
        s = " "
        for i in self.map.values():
            s += str(i)
            s += "\n\t  "
        return f"GeneSet {{{s}}}"