import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Any
import webbrowser

from neuron.optimizer.neat.core import Neat
from neuron.optimizer.neat.genome import Genome
from neuron.core.params_loader import GENERATIONS, POPULATION_SIZE

plt.ion()
fig, ax = plt.subplots(4, 1, sharex='all')


class SuperScenario:
    def __init__(self, file_name: str = None, neat: Neat = None):
        self.file_name = file_name
        self.neat = neat
        self.population = self.neat.generate_population(POPULATION_SIZE, self.fitness_function)

    @staticmethod
    def fitness_function(genome: Genome, visualize: bool = False, f_fig=None, f_ax=None, *args, **kwargs) -> Union[float, Any, None]:
        raise NotImplementedError("Can't call fitness_function from a SuperScenario")

    def visualize(self, block: bool = True, *args, **kwargs) -> None:
        global fig, ax
        genome: Genome = Genome.load(self.file_name)

        if not block:
            fig, ax = self.fitness_function(genome, visualize=True, f_fig=fig, f_ax=ax, *args, **kwargs)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.2)
        else:
            plt.ioff()
            fig, ax = self.fitness_function(genome, visualize=True, f_fig=fig, f_ax=ax, *args, **kwargs)
            plt.show()

    def visualize_save(self, block: bool = True, *args, **kwargs) -> None:
        self.visualize(False)
        plt.savefig(f"{self.file_name}.png")
        plt.savefig(f"{self.file_name}.eps")

    def visualize_genome_and_save(self, *args, **kwargs):
        genome: Genome = Genome.load(self.file_name)
        genome.visualize(self.file_name)
        # webbrowser.open_new(f"{self.file_name}.html")

    def run(self) -> None:
        # Print Start Message
        print(f"Starting Neat with population of {POPULATION_SIZE} for {GENERATIONS} generations")

        convergence: list[float] = []

        for i in range(GENERATIONS):
            # update population
            self.population.update(i == GENERATIONS, save_file_name=self.file_name)
            convergence.append(self.population.best_fitness)
            # print updates
            s = f"----- Generation {i + 1} -----\n"
            s += f"Generation {i + 1} Best = {self.population.best_fitness}\n"
            s += f"Generation {i + 1} Worst = {self.population.worst_fitness}\n"
            print(s)
            self.visualize(False)
            plt.savefig(f"{self.file_name}/generation_{i}.png")

        plt.savefig(f"{self.file_name}.eps")
        plt.savefig(f"{self.file_name}.png")

        plt.ioff()
        plt.figure().clear()
        plt.plot(np.arange(1, len(convergence) + 1, 1), convergence)
        plt.xlabel('generation')
        plt.ylabel('fitness')
        plt.savefig(f"{self.file_name}_convergence_curve.png")
        plt.show()

    def test(self):
        try:
            self.run()
        except KeyboardInterrupt:
            print("STOP Evolving")
            print("Saving and Exiting")
        except Exception as e:
            print(e)
