import importlib
from scenarios.core import SuperScenario
from timeit import timeit as timer
from datetime import timedelta

if __name__ == "__main__":
    # --------------------- TEST NUMBER ---------------------
    TEST = '01'  # Scenarios Not Working (4,6,7,8)

    # --------------------- IMPORT SCENARIO ---------------------
    scenarioModule = importlib.import_module(f'scenarios.scenario_{TEST}.scenario')
    scenario: SuperScenario = scenarioModule.Scenario()
    # start = timer()
    # --------------------- START TESTING ---------------------
    scenario.visualize()
    # print(timedelta(seconds=timer()-start))

# import numpy as np
# from math import sin, pi
# from neuron.core.neuro_controller_2 import NeuroController
# from neuron.core.params_loader import TIME_STEP
# import matplotlib.pyplot as plt
# def main():
#     t = np.arange(0, 0.1, 0.0005)
#     n = NeuroController([[0,0.8],[0,0]],[0],[1],TIME_STEP)

#     v: list[list[float], list[float], list[float]] = [[], []]
    
#     for i in range(len(t)):
#         n.step([0.1], 0 , 0)
#         if n.s[0][0] == 1:
#             v[0].append(t[i])
#         if n.s[1][0] == 1:
#             v[1].append(t[i])

#     _, ax = plt.subplots(2, 1, sharex='all')
#     ax[0].eventplot(v[0], colors=(0, 1, 0))
#     # ax[0].grid()
#     ax[0].eventplot(v[1], colors=(1, 0, 0))
#     ax[0].grid()
#     plt.show()


# if __name__ == '__main__':
#     main()

