from math import sin, cos, pi
# import numpy as np
# from plotly.subplots import make_subplots # type: ignore
# import plotly.graph_objects as go # type: ignore

class BiCopter:
    def __init__(self, m=1.0, g=9.81, theta=0.0) -> None:
        self.m = m
        self.g = g
        self.M1 = 0.25
        self.M2 = 0.25
        self.theta = theta
        self.theta_dot = 0.0
        self.a = 0.1
        self.b = 0.1
        self.l = 1.0

    def step(self, w1, w2, t, dt):
        self.theta_dot += dt*(((self.M2 - self.M1)*self.g*(self.l/2)*cos(self.theta)-self.b*self.theta_dot - self.a *
                               self.theta_dot*(w2 + w1) + 0.04*self.l*(w2-w1))/((self.l**2)/4)*(self.M1+self.M2 + self.m/3))
        self.theta += dt*self.theta_dot

        return self.theta, self.theta_dot


# if __name__ == '__main__':
#     SAMPLES = 10000
#     b = BiCopter()
#     v1 = [0.0]*SAMPLES
#     v2 = [0.0] * SAMPLES
#     v3 = [0.0] * SAMPLES
#
#     for i in range(SAMPLES):
#         v1[i], v2[i] = b.step(0.25, 0.1, 0, 0.0005)
#         v3[i] = i*0.0005
#
#     fig = make_subplots(rows=2,
#                         cols=1,
#                         shared_xaxes=True,
#                         vertical_spacing=0.02,
#                         x_title="t(s)")
#
#     fig.add_trace(
#         go.Scatter(x=v3, y=v1, name='theta'),
#         row=1, col=1
#     )
#
#     fig.add_trace(
#         go.Scatter(x=v3, y=v2, name='theta dot'),
#         row=2, col=1
#     )
#     fig.update_layout(height=720, width=1080)
#     # fig.show()
#     fig.write_image("Hello.png")
