import importlib
from scenarios.core import SuperScenario

if __name__ == "__main__":
    # --------------------- TEST NUMBER ---------------------
    TEST = '01'  # Scenarios Not Working (4,6,7,8)

    # --------------------- IMPORT SCENARIO ---------------------
    scenarioModule = importlib.import_module(f'scenarios.scenario_{TEST}.scenario')
    scenario: SuperScenario = scenarioModule.Scenario()

    # --------------------- START TESTING ---------------------
    scenario.test()

# import numpy as np
# from neuron.core.neuro_controller import NeuroController as N1
# from neuron.core.neuro_controller_2 import NeuroController as N2
# from neuron.core.params_loader import TIME_STEP
# import matplotlib.pyplot as plt
#
# def main():
#     t = np.arange(0, 0.1, 0.0005)
#     n1 = N1([[0, 1, 1, 0], [0, 0, 0, -1], [0, 0, 0, 1], [0, 0, 0, 0]], [0], [3], TIME_STEP)
#     n2 = N2([[0, 1, 1, 0], [0, 0, 0, -1], [0, 0, 0, 1], [0, 0, 0, 0]], [0], [3], TIME_STEP)
#
#     v: list[list[float], list[float], list[float]] = [[], []]
#
#     for i in range(len(t)):
#         n1.step([1], TIME_STEP * i, TIME_STEP)
#         n2.step([1], TIME_STEP * i, TIME_STEP)
#
#         # v[0].append(n1.neurons[0].s)
#         # v[1].append(n2.s[0][0])
#
#         if n1.neurons[1].s:
#             v[0].append(t[i])
#         if n2.s[1][0]:
#             v[1].append(t[i])
#
#     # plt.plot(t, v[0])
#     # plt.plot(t, v[1])
#     _, ax = plt.subplots(2, 1, sharex=True)
#     ax[0].eventplot(v[0], colors=(0, 1, 0))
#     ax[0].grid()
#     ax[1].eventplot(v[1], colors=(1, 0, 0))
#     ax[1].grid()
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
