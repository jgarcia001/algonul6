import numpy as np
import matplotlib.pyplot as plt

import methodes as m

N = 3000
h = 30


birth = 200
death = 150
limit_people = 500

def derivative_function_without_limit():
    return (lambda x,t: (birth - death) * x)

def derivative_function_limit():
    return (lambda x,t: (birth - death) * x * (1 - (x/limit_people)))

def variation_population_function(population_zero, time_zero, derivative):
    return m.meth_n_step(population_zero, time_zero, N, h, derivative(), m.step_runge_kutta_4)

population_zero = 10
time_zero = 0
times = np.arange(0, N, 1)

population_array = variation_population_function(population_zero, time_zero, derivative_function_without_limit)


plt.plot(times, population_array)

population_array_limit = variation_population_function(population_zero, time_zero, derivative_function_limit)

plt.plot(times, population_array_limit)
plt.show()
