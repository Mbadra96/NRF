import os
import json
from neuron.utils.units import sec, ms

from numpy import arange, ndarray

f = open(os.getcwd()+"/params.json", 'rb')
params = json.load(f)
f.close()

# --------------- TIME ----------------
POPULATION_SIZE = params['PopulationNumber']
GENERATIONS = params['Generation']
TIME:float = params["Evaluation_Time"] * sec
TIME_STEP:float = params["Time_Step"] * ms  # dt is 0.5 ms
SAMPLES:int = int(TIME / TIME_STEP)
t:ndarray = arange(0, TIME, TIME_STEP)

# --------------- SPECIES ----------------
SPECIES_THRESHOLD = params["Species_threshold"]
SPECIES_ELIMINATION_RATE = params["Species_elimination_rate"]

# --------------- POPULATION ----------------
POPULATION_CROSSOVER_RATE = params["Population_crossover_rate"]
POPULATION_MUTATION_RATE = params["Population_mutation_rate"]

# --------------- Neuron Dynamics----------------

REFRACTORY_PERIOD: float = params['RP'] * ms
TN: float = params['Tn'] * ms
C_SYN: float = params['Csyn']
R: float = params['R']
