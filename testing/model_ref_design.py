import matplotlib.pyplot as plt
import numpy as np
from math import exp, sin, cos


# The is the Model Reference control Design based in Two parameters which is Overshoot and settling time

def reference_model(time: float) -> float:
    return 1 - exp(-4 * time / 3) * (cos(1.819168472 * time) + 0.7329 * sin(1.819168472 * time))


def reference_model_dot(time: float) -> float:
    return 2.796415937 * exp(-4 * time / 3) * sin(1.819168472 * time)


def main() -> None:
    t = np.arange(0, 10, 0.001)
    v = [0.0] * len(t)
    v_dot = [0.0] * len(t)

    for i in range(1, len(t)):
        v[i] = reference_model(t[i])
        v_dot[i] = reference_model_dot(t[i])

    plt.subplot(2, 1, 1)
    plt.plot(t, v)
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(t, v_dot)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
