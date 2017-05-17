#===============IMPORTS===============#

import numpy.linalg as nl

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
    for i in range(N):
        start_y = meth(start_y, start_x, h, f)
        start_x += h
    return start_y

def meth__epsilon(y0, t0, tf, esp, f, meth):
    start_y = y0
    start_x = t0
    while (nl.norm(meth(start_y, start_x, h, f) - f(start_y, start_x)) > eps):
        start_y = meth(start_y, start_x, h, f)
        start_x += h
    return start_y

#==============TANGENTS===============#

def tangents_field(y, t, h, f):
    tan_t = lambda x : f(y, t)*(x - t) + y
    

#==============TEST_ZONE==============#

def test_methodes():
    return
