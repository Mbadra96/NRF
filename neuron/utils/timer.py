import time


class Timer:
    """
        This class is used to calculate the time of a specific scope:

        timer = Timer("NAME")

        function_to_measure()

        timer.stop()
    """

    def __init__(self, name: str = ""):
        self.__start = time.perf_counter()
        self.__name = name
        self.end = None

    def stop(self):
        self.end = time.perf_counter()

    def __del__(self):
        print(f"{self.__name} time duration is = {(self.end - self.__start)*1000} ms")


