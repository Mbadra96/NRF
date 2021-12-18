from math import sin, cos


class InvertedPendulum:
    def __init__(self, m1=0.5, m2=0.2, length=0.3, theta_0=0) -> None:
        self.M = m1
        self.m = m2
        self.Mm = self.M + self.m
        self.b1 = 0.2
        self.b2 = 0.002
        self.I = 0.006
        self.x = 0
        self.x_d = 0
        self.x_dd = 0
        self.theta = theta_0
        self.theta_d = 0
        self.theta_dd = 0
        self.l = length

    def step(self, f, t, dt):
        self.x_dd = - ((self.m*self.l)/self.Mm) * self.theta_dd * cos(self.theta) - ((self.m*self.l)/self.Mm)*(self.theta_d**2)*sin(self.theta)- (self.b1/self.Mm)*self.x_d + (f/self.Mm)
        self.x_d += dt * self.x_dd
        self.x += dt * self.x_d

        self.theta_dd = -((self.m*self.l)/self.I+self.m*self.l**2)*self.x_dd*cos(self.theta)- (self.b2/(self.I+self.m*self.l**2))*self.theta_d - ((self.m*9.81*self.l)/(self.I+self.m*self.l**2))*sin(self.theta)

        self.theta_d += dt * self.theta_dd
        self.theta += dt * self.theta_d

        return self.theta, self.theta_d, self.x, self.x_d


