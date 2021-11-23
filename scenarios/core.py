from neuron.optimizer.neat.genome import Genome
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots(4, 1, sharex=True)

class SuperScenario:

    def run(self) -> None:
        raise NotImplementedError

    def visualize(self, block: bool = True) -> None:
        global fig, ax
        genome: Genome = Genome.load(self.file_name)
        # fig, ax = self.fitness_function(genome, visualize=True, fig=fig, ax=ax)

        if not block:
            fig, ax = self.fitness_function(genome, visualize=True, fig=fig, ax=ax)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.2)
        else:
            plt.ioff()
            fig, ax = self.fitness_function(genome, visualize=True, fig=fig, ax=ax)
            plt.show()
            input("Press Enter")

    def visualize_and_save(self) -> None:
        raise NotImplementedError

    def test(self):
        try:
            self.run()
        except KeyboardInterrupt:
            print("STOP Evolving")
            print("Saving and Exiting")
        except Exception as e:
            print(e)
        finally:
            self.visualize_and_save()
