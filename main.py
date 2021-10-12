from scenarios.scenario_01.scenario import Scenario01


if __name__ == "__main__":
    scenario = Scenario01()
    scenario.run()
    scenario.visualize_and_save()
    