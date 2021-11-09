import os
import json
from neuron.utils.units import sec, ms

from numpy import arange

f = open(os.getcwd()+"/params.json",'rb')
params = json.load(f)
f.close()

# --------------- TIME ----------------
POPULATION_SIZE = params['PopulationNumber']
GENERATIONS = params['Generation']
TIME = params["Evaluation_Time"] * sec
TIMESTEP = params["Time_Step"] * ms  # dt is 0.5 ms
SAMPLES = int(TIME / TIMESTEP)
t = arange(0, TIME, TIMESTEP)

# --------------- SPECIES ----------------
SPECIES_THRESHOLD = params["Species_threshold"]
SPECIES_ELIMINATION_RATE = params["Species_elimination_rate"]

# --------------- POPULATION ----------------
POPULATION_CROSSOVER_RATE = params["Population_crossover_rate"]
POPULATION_MUTATION_RATE = params["Population_mutation_rate"]


