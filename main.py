from scenarios.scenario_08.scenario import Scenario08

if __name__ == "__main__":
    scenario = Scenario08()
    try:
        scenario.run()
    except KeyboardInterrupt:
        print("STOP Evolving")
        print("Saving and Exiting")

    finally:
        scenario.visualize_and_save()
    