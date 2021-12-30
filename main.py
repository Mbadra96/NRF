import importlib
from scenarios.core import SuperScenario

if __name__ == "__main__":
    # --------------------- TEST NUMBER ---------------------
    TEST = '18'  # Scenarios Not Working (4,6,7,8)

    # --------------------- IMPORT SCENARIO ---------------------
    scenarioModule = importlib.import_module(f'scenarios.scenario_{TEST}.scenario')
    scenario: SuperScenario = scenarioModule.Scenario()  # type: ignore

    # --------------------- START TESTING ---------------------
    scenario.visualize_genome_and_save()
