from neuron.utils.units import *
from neuron.simulation.levitating_ball import LevitatingBall
import numpy as np
import matplotlib.pyplot as plt


# In Conclusion of the error hassle it is notices that the error in the PID should be the difference in the heights only
# an it works with some modifications to the (KI, KP, KD) terms.
# but adding the velocity error to the error function shows more stability to the control input function for some reason
# which should be more logical in my point of view as it should give control to the velocity state in addition to
# the position state.

TIME = 10 * sec
TIMESTEP = 0.5 * ms # dt is 0.5 ms
SAMPLES = int(TIME/TIMESTEP)

t = np.arange(0, TIME, TIMESTEP)
v1 = [0.0]*SAMPLES
v2 = [0.0]*SAMPLES
v3 = [0.0]*SAMPLES

print(f"Simulation of Neuron Pool for TIME = {TIME} sec and TIMESTEP = {TIMESTEP} ms with SAMPLES = {SAMPLES}")

F = 0.0
kp = 10
ki = 14
kd = 0.1
x_ref = 1
x_dot_ref = 0
e_I = 0.0
e_last = 0.0
e_dot = 0.0

if __name__ == "__main__":

    ball = LevitatingBall(1, 0, 0)

    for i in range(SAMPLES):
        v1[i], v2[i] = ball.step(F, t[i], TIMESTEP)

        # Get Errors
        e = (x_ref - v1[i]) + (x_dot_ref - v2[i])
        e_I = e_I + e * TIMESTEP
        e_dot = (e - e_last) / TIMESTEP
        # Update Error
        e_last = e

        # Get Force
        F = (kp * e) + (ki * e_I) + (kd * e_dot)
        v3[i] = F

    plt.subplot(3, 1, 1)
    plt.plot(t, v1)
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t, v2)
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t, v3)
    plt.grid()

    plt.show()
    