from solver import  Euler,RK4 # type: ignore
import matplotlib.pyplot as plt # type: ignore

global Vr, R, C, Tn

Vr = 0
R = 10
C = 0.1
Tn = R*C



def dvdt(v,_,I):
    return (-(v-Vr) + R*I)/Tn

if __name__ == "__main__":
    
    v = [0]*101
    t = [0]*101

    for i in range(100):
        v[i+1],t[i+1] = RK4.step(dvdt,v[i],t[i],0.01,1)
    
    print(f" {v[-1]}, {t[-1]} ")
    plt.plot(t,v)
    plt.grid()
    plt.show()
    
