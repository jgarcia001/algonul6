import numpy as np
import matplotlib.pyplot as plt

import methodes as m

N = 10
h = 0.01

a = 50
b = 2
c = 20
d = 4

def derivative_function():
    return (lambda x,t: np.array(x[0] * (a - b * x[1]), x[1] * (c * x[0] - d))) 

def prey_predator_function(prey_predator_zero, time_zero, derivative):
    return m.meth_n_step(prey_predator_zero, time_zero, N, h, derivative(), m.step_runge_kutta_4)

prey_zero = 10
predator_zero = 20
prey_predator_zero = np.array([prey_zero, predator_zero])

time_zero = 0
times = np.arange(0, N, 1)

prey_predator_array = prey_predator_function(prey_predator_zero, time_zero, derivative_function)

prey_array = np.empty(N)
predator_array = np.empty(N)
for i in range(0, N, 1):
    prey_array[i] = prey_predator_array[i][0]
    predator_array[i] = prey_predator_array[i][1]
    
plt.plot(times, prey_array)
plt.plot(times, predator_array)

plt.show()

plt.plot(prey_array, predator_array)

plt.show()
