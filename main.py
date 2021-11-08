from scenarios.scenario_03.scenario import Scenario03

if __name__ == "__main__":
    scenario = Scenario03()
    try:
        scenario.run()
    except KeyboardInterrupt:
        print("STOP Evolving")
        print("Saving and Exiting")
    finally:
        scenario.visualize_and_save()
    