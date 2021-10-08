class Euler:    
    @staticmethod 
    def step(func,x,t,dt,*args, **kwargs):
        x += func(x, t, *args, **kwargs) * dt
        return x, t 

class RK4:
    @staticmethod        
    def step(func,x,t,dt,*args, **kwargs):
        dx1 = func(x,t, *args, **kwargs) * dt
        dx2 = func(x + dx1 * 0.5,t + dt * 0.5, *args, **kwargs) * dt
        dx3 = func(x + dx2 * 0.5,t + dt * 0.5, *args, **kwargs) * dt
        dx4 = func(x + dx3,t + dt , *args, **kwargs) * dt
        x += 1 / 6 * (dx1 + dx2 * 2 + dx3 * 2 + dx4)
        return x, t