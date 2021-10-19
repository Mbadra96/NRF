from scenarios.scenario_05.scenario import Scenario05

if __name__ == "__main__":
    scenario = Scenario05()
    try:
        scenario.run()
    except KeyboardInterrupt:
        print("STOP Evolving")
        print("Saving and Exiting")

    finally:
        scenario.visualize_and_save()
    