import importlib
from scenarios.core import SuperScenario

if __name__ == "__main__":
    # --------------------- TEST NUMBER ---------------------
    TEST = "19"

    # --------------------- IMPORT SCENARIO ---------------------
    scenarioModule = importlib.import_module(f'scenarios.scenario_{TEST}.scenario')
    scenario: SuperScenario = scenarioModule.Scenario()  # type: ignore

    # --------------------- START TESTING ---------------------
    scenario.visualize(m2=0.202)
