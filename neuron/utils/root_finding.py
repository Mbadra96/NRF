# Defining Function
import math
R = 10
CSYN = 2
W = 1
TS = 10**(-3)
TM = 10**(-2)

import numpy as np

def ISyn(t):
    return CSYN*R*math.exp(-t/TS)

def func(x):
    return ((R*CSYN*W*TS )/(TM-TS))*(math.exp(-x/TM) - math.exp(-x/TS)) - 1

# Implementing False Position Method
def false_position(f,x0,x1,e):
    step = 1
    print('\n*** FALSE POSITION METHOD IMPLEMENTATION ***')
    condition = (f(x0) * f(x1) < 0)
    if not condition:
        print(f"condition is not met")
        return None
    while condition and abs(f(x1)-f(x0)) > e:
        x2 = x0 - (x1-x0) * f(x0)/( f(x1) - f(x0) )
        print('Iteration-%d, x2 = %0.10f and f(x2) = %0.10f' % (step, x2, f(x2)))

        if f(x0) * f(x2) < 0:
            x1 = x2
        else:
            x0 = x2

        step = step + 1
        condition = (f(x0) * f(x1) < 0)

    return x1, f(x1)
    print('\nRequired root is: %0.8f' % x2)
    
from os import system
clear = lambda: system('clear')
import matplotlib.pyplot as plt # type: ignore
if __name__ == "__main__":
    clear()
    # print(func(0))
    # print(func(0.5))
    bounds = (0.005,0.01)
    try:
        t1,f =false_position(func,*bounds,0.0000001)   
    except:
        pass
    

    t = np.arange(0,0.1,0.001)
    v = [0]*len(t)
    for i in range(len(t)):
        v[i] = func(t[i])
    plt.plot(t,v) 
    plt.plot([bounds[0],bounds[0]],[-1,1])
    plt.plot([bounds[1],bounds[1]],[-1,1])    
    try:
        if t1:
            plt.plot(t1,f,'o')
            plt.plot([t1,t1],[-1,f])   
            plt.text(0.05,0.5,f"t = {t1}",fontsize=12)
            
    except:
        pass
    
    plt.grid()
    plt.show()
    