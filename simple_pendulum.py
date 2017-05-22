import numpy as np
import matplotlib.pyplot as plt

import methodes as m

N = 100
h = 0.1 

g = 9.81
length = 0.5
theta_init = 10.0

def f_function():
    
    return (lambda x,t: np.array([x[1], ((g/length) * -(np.sin(x[0])))]))
    
def pendulum_position_function(y_zero):

    f = f_function()
    return m.meth_n_step(y_zero, 0, N, h, f, m.step_runge_kutta_4)

def find_period(y_array):
    index_start_zero = 0
    index_end_zero = 0
    count_zero = -1
    num_period = 0

    n = y_array.size
    i = 1
    while (count_zero == -1) and i < n:
        if((y_array[i] * y_array[i - 1]) < 0):
            index_start_zero = i
            count_zero = 0
        i = i + 1
        
    if (not (count_zero == 0)):
        return -1
    
    while i < n:
        if((y_array[i] * y_array[i - 1]) < 0):
            count_zero = count_zero + 1
            if(count_zero == 2):
                count_zero = 0
                num_period = num_period + 1
                index_end_zero = i
        i = i + 1

    if (num_period == 0):
        return -1
    
    return (index_end_zero - index_start_zero) / num_period 



y_zero = np.empty(2)
y_array = np.empty(N)
number_angle = 89
frequency = np.empty(number_angle)
for j in range (0, number_angle, 1):
    y_zero[0] = np.radians(j + 1)
    y_zero[1] = 0

    y_array[0] = y_zero[0]
    y = pendulum_position_function(y_zero)
    for i in range (1, N):
        y_array[i] = np.degrees(y[i][0])

    period = find_period(y_array)
    print(period)
    frequency[j] = 1/period
print(frequency)
times = np.arange(0, number_angle, 1)
plt.plot(times, frequency)
frequency_little_angle = (np.sqrt(g/length)/(np.pi*2))*h
print(1/frequency_little_angle)
plt.plot(times, np.full((number_angle),frequency_little_angle))
plt.show()
