from neuron.optimizer.neat.species import Member, Species
from neuron.optimizer.neat.genome import Genome

class Population:
    def __init__(self,size:'int',evaluation_function) -> None:
        self.population = []
        self.species = []
        self.size = size
        self.evaluation_function = evaluation_function

    def get(self,index:'int')->'Member':
        return self.population[index]

    def add_genome(self,genome:'Genome') -> None:
        self.population.append(Member(genome,-1.0))

    def get_species_size(self):
        return len(self.species)

    def sort(self):
        # Assending order
        n = len(self.population)
        for i in range(n-1):
            for j in range(0,n-i-1):
                if self.population[j].fitness > self.population[j+1].fitness:
                    self.population[j], self.population[j+1] = self.population[j+1], self.population[j]

    def print_fitness(self,generation_number):
        print(f"-----Generation {generation_number} -------------")
        print(f"Generation {generation_number} Best = {self.population[0].fitness}")
        print(f"Generation {generation_number} Worst = {self.population[-1].fitness}")

    def evolve(self):
        self.population.clear()
        species_share = self.size/len(self.species)

        for species in self.species:
            for i in range(int(species_share)):
                self.population.append(Member(species.reproduce(),-1.0))

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

    def update(self,generation_number:'int',is_last:bool):
        # Evaluate all members in population
        
        for member in self.population:
            member.fitness = self.evaluation_function(member.genome)
        
        self.sort()
        self.print_fitness(generation_number)
        if not is_last:
            self.update_species()
            self.evolve()

    def stop(self):
        for member in self.population:
            member.fitness = self.evaluation_function(member.genome)
        

