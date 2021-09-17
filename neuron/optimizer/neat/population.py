from numpy import Inf
from numpy.core.arrayprint import set_printoptions
from neuron.optimizer.neat.species import Member, Species
from neuron.optimizer.neat.genome import Genome

class Population:
    def __init__(self,size:'int',evaluation_function) -> None:
        self.population = []
        self.species = []
        self.size = size
        self.evaluation_function = evaluation_function
        self.best_genome = None

    def get(self, index: 'int') -> 'Member':
        return self.population[index]

    def add_genome(self, genome: 'Genome') -> None:
        self.population.append(Member(genome, Inf))

    def get_species_size(self):
        return len(self.species)

    def sort(self):
        # Ascending order
        n = len(self.population)
        for i in range(n-1):
            for j in range(0,n-i-1):
                if self.population[j].fitness > self.population[j+1].fitness:
                    self.population[j], self.population[j+1] = self.population[j+1], self.population[j]

    def print_fitness(self, generation_number):
        print(f"----- Generation {generation_number} -----")
        best_fitness = Inf
        worst_fitness = 0
        print("Species => ", end="[ ")
        for species in self.species:
            print(len(species.members), end=" ")

            if best_fitness > species.members[0].fitness:
                best_fitness = species.members[0].fitness
                self.best_genome = species.members[0]
            if worst_fitness < species.members[-1].fitness:
                worst_fitness = species.members[-1].fitness
        print("]")
        print(f"Generation {generation_number} Best = {best_fitness}")
        print(f"Generation {generation_number} Worst = {worst_fitness}")

    def evolve(self):
        self.population.clear()
        species_share = self.size/len(self.species)

        for species in self.species:
            for i in range(int(species_share)):
                self.population.append(Member(species.reproduce(), Inf))

    def update_species(self):
        for member in self.population:
            compatible = False
            for species in self.species:
                if species.add(member):
                    compatible = True
                    break
            if not compatible:
                self.species.append(Species(member))
                
        for species in self.species:
            species.sort()
            species.eliminate()

    def update(self, generation_number: 'int', is_last: bool, save_best: bool=False):
        # Evaluate all members in population
        
        for member in self.population:
            member.fitness = self.evaluation_function(member.genome)
        
        self.sort()
        self.update_species()

        if not is_last:
            self.evolve()
        self.print_fitness(generation_number)
        self.best_genome.genome.save("best")
