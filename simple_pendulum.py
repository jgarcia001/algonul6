import numpy as np
import matplotlib.pyplot as plt

import methodes as m

N = 1
h = 1

g = 9.81
length = 0.5
theta_init = 80

def f_function():
    
    return (lambda x,t: np.array([x[1], (-g/length) * np.sin(x[0])]))
    
def frequency_function(y_zero):

    f = f_function()
    return m.meth_n_step(y_zero, 0, N, h, f, m.step_middle_point)

    
y_zero = np.empty(2)
y_zero[0] = theta_init
y_zero[1] = 0
times = np.arange(0, 20, 1)
y_array = np.empty(times.size)
y_array[0] = y_zero[0]

for i in range (1, times.size):
    y = frequency_function(y_zero)
    y_array[i] = y(i)

plt.plot(times, y_array)
plt.show()
