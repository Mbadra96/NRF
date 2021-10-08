from neuron.utils.randomizer import Randomizer
from neuron.optimizer.neat.genome import Genome

class Member:
    def __init__(self,genome:'Genome',fitness:float=0.0) -> None:
        self.genome = genome
        self.fitness = fitness

class Species:
    THRESHOLD = 1.0
    ELIMNATIONRATE = 0.7

    def __init__(self,represntative:'Member') -> None:
        self.members:list[Member] = []
        self.represntative = represntative
        self.add(represntative)

    def add(self,candidate:'Member') -> 'bool':
        if self.is_compatible(candidate):
            self.members.append(candidate)
            return True
        return False

    def is_compatible(self,candidate:'Member')->'bool':
        if self.represntative.genome.get_distance(candidate.genome) <= Species.THRESHOLD:
            return True
        return False

    def sort(self):
        # Assending order
        n = len(self.members)
        for i in range(n-1):
            for j in range(0,n-i-1):
                if self.members[j].fitness > self.members[j+1].fitness:
                    self.members[j], self.members[j+1] = self.members[j+1], self.members[j]
        # make the best performing member as represntative 
        # self.represntative = self.members[0]

        
    def reproduce(self)->'Genome':
        if len(self.members) <= 1:
            return self.represntative.genome.get_mutated_child()

        g1 = Randomizer.choice(self.members).genome
        g2 = Randomizer.choice(self.members).genome
       
        while g1 == g2:
            g2 = Randomizer.choice(self.members).genome

        return g1.crossover(g2).mutate()

    def eliminate(self):
        if int(Species.ELIMNATIONRATE*len(self.members)) > 1:
            self.members = self.members[0:int(Species.ELIMNATIONRATE*len(self.members))]

    def __str__(self) -> str:
        s = ""
        for i,member in enumerate(self.members):
            s += f"member {i} : fitness = {member.fitness} \n"

        return s
    
