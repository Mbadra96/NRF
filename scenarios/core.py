from neuron.optimizer.neat.genome import Genome
import matplotlib.pyplot as plt


class SuperScenario:

    def run(self) -> None:
        raise NotImplementedError

    def visualize(self) -> None:

        genome: Genome = Genome.load(self.file_name)

        self.fitness_function(genome, visualize=True)

        plt.show()

    def visualize_and_save(self) -> None:
        raise NotImplementedError

    def test(self):
        try:
            self.run()
        except KeyboardInterrupt:
            print("STOP Evolving")
            print("Saving and Exiting")
        finally:
            self.visualize_and_save()
