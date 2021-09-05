
class LevitatingBall:
    def __init__(self, mass, x_0, x_dot_0) -> None:
        self.mass = mass
        self.x = x_0
        self.x_dot = x_dot_0

    def step(self, i, t, dt):

        self.x_dot += dt*(i-self.mass * 9.81)/self.mass
        self.x += dt*self.x_dot
        
        if self.x <= 0:
            self.x = 0
            self.x_dot = 0

        elif self.x >= 10:
            self.x = 10
            self.x_dot = 0
            
        return self.x, self.x_dot

    def evaluate_controller(self, controller) -> float:
        pass
        
    