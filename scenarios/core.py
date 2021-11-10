class SuperScenario:

    def run(self) -> None:
        raise NotImplementedError

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
