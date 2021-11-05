from scenarios.scenario_02.scenario import Scenario02

if __name__ == "__main__":
    scenario = Scenario02()
    # try:
    #     scenario.run()
    # except KeyboardInterrupt:
    #     print("STOP Evolving")
    #     print("Saving and Exiting")
    #
    # finally:
    scenario.visualize_and_save()
    