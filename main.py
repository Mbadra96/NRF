from scenarios.scenario_01.scenario import Scenario01

if __name__ == "__main__":
    scenario = Scenario01()
    try:
        scenario.run()
    except KeyboardInterrupt:
        print("STOP Evolving")
        print("Saving and Exiting")
    finally:
        scenario.visualize_and_save()
    