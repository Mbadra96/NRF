class Motor:

    def __init__(self, L=0.5, R=1, b=0.1, J=0.01, K=0.01, i_0=0, w_0=0) -> None:
        self.L = L
        self.R = R
        self.b = b
        self.J = J
        self.K = K
        self.i = i_0
        self.w = w_0

    def step(self, v, t, dt):

        self.i += dt * (- self.K * self.w - self.R * self.i + v) / self.L
        self.w += dt * (-self.b * self.w + self.K * self.i)/self.K

        return self.w
