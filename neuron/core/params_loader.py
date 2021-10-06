import os
import json

f = open(os.getcwd()+"/params.json",)
params = json.load(f)
f.close()
