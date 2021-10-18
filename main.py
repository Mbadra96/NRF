from scenarios.scenario_04.scenario import Scenario04

if __name__ == "__main__":
    scenario = Scenario04()
    try:
        scenario.run()
    except KeyboardInterrupt:
        print("STOP Evolving")
        print("Saving and Exiting")

    finally:
        scenario.visualize_and_save()
    