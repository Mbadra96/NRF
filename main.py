import importlib

if __name__ == "__main__":
    # --------------------- TEST NUMBER ---------------------
    TEST = '01'

    # --------------------- IMPORT SCENARIO ---------------------
    scenario = importlib.import_module(f'scenarios.scenario_{TEST}.scenario')

    # --------------------- START TESTING ---------------------
    scenario.Scenario().test()
