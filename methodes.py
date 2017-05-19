#===============IMPORTS===============#

import numpy.linalg as nl
import matplotlib.pyplot as plt
import numpy as np

#=================CORE================#

# Euler
def step_euler(y, t, h, f):
    return y + h * f(y,t)

# Middle point
def step_middle_point(y, t, h, f):
    return y + h * f(y + h/2*f(y,t), t + h/2)

# Heun
def step_heun(y, t, h, f):
    return y + h/2 * (f(y,t) + f(y + h*f(y,t) ,t + h))

# Runge-Kutta order 4
def step_runge_kutta_4(y, t, h, f):
    p1 = f(y,t)
    p2 = f(y + p1*h/2,t + h/2)
    p3 = f(y + p2*h/2,t + h/2)
    p4 = f(y + h*p3, t + h)
    p = (p1 + 2*p2 + 2*p3 + p4)/6
    return y + h * p

# Generalisation
def meth_n_step(y0, t0, N, h, f, meth):
    start_y = y0
    start_x = t0
    Y = [start_y]
    for i in range(N):
        start_y = meth(start_y, start_x, h, f)
        start_x += h
        Y.append(start_y)
    return Y

print(meth_n_step(0., 1., 100, 0.01, lambda y,t : y/(1 + t**2), step_euler))

def meth__epsilon(y0, t0, tf, esp, f, meth):
    start_y = y0
    start_x = t0
    Y = [start_y]
    while (nl.norm(meth(start_y, start_x, h, f) - f(start_y, start_x)) > eps):
        start_y = meth(start_y, start_x, h, f)
        start_x += h
        Y.append(start_y)
    return Y

#==============TANGENTS===============#

def tangents_field(y0, t0, h, f, meth, N):
    tan0 = lambda x : f(y0, t0).dot((x - t0)) + y0
    X = [0.]*N
    Y = [0.]*N
    DY = [0.]*N
    X[0] = t0
    Y[0] = y0
    DY[0] = f(y0, t0)
    for i in range(1,N):
        y0 = meth(y0, t0, h, f)
        t0 += h
        X[i] = t0
        Y[i] = y0
        DY[i] = f(y0, t0)
    plt.quiver(X, DY)
    plt.show()
    return

#tangents_field(1, 0, 0.01, lambda y,t : y/(1 + t**2), step_euler, 50)

#==============TEST_ZONE==============#

def test_methodes():
    # Dimension 1
    t0 = 0
    y0 = 1
    f = lambda y,t : y/(1 + t**2)
    return
