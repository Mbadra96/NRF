from neuron.utils.units import *
from neuron.simulation.levitating_ball import LevitatingBall
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from math import sin,pi

TIME = 5 * sec
TIMESTEP = 0.5 * ms # dt is 0.1 ms
SAMPLES = int(TIME/TIMESTEP)

t = np.arange(0,TIME,TIMESTEP)
v1 = [0]*SAMPLES
v2 = [0]*SAMPLES
v3 = [0]*SAMPLES

print(f"Simulation of Neuron Pool for TIME = {TIME} sec and TIMESTEP = {TIMESTEP} ms with SAMPLES = {SAMPLES}")

F = 0
kp = 20
ki = 10
kd = 0.1
x_ref = 2
x_dot_ref = 0
e_I = 0
e_last = 0
e_dot = 0
total_error = 0
if __name__ == "__main__":
    ball = LevitatingBall(1,0,0)
    total_F = 0
    for i in range(SAMPLES):    
        v1[i],v2[i]=ball.step(F,t[i],TIMESTEP)
        e = (x_ref - v1[i]) + (x_dot_ref - v2[i])
        e_I += e
        total_error += abs((x_ref - v1[i])/10.0) 
        v3[i] = e
        e_dot = (e - e_last)/TIMESTEP
        F = (kp * e )+ (ki * e_I) + (kd * e_dot)
        total_F += abs(F/20)
        e_last = e
        
    print(total_error+0.01*total_F)
    
    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(
        go.Scatter(x=t, y=v1),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=t, y=v2),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=t, y=v3),
        row=3, col=1
    )
    fig.update_layout(height=720, width=1080, title_text="Side By Side Subplots")
    fig.show()
    