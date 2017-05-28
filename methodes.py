#===============IMPORTS===============#

import numpy.linalg as nl
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as sm

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
    #if (type(y0) is float or type(y0) is int or type(y0) is complex):
        #Y = np.empty(N)
        #Y = np.empty([N, y0.size])
    Y = [start_y]
    for i in range(1,N):
        start_y = meth(start_y, start_x, h, f)
        start_x += h
        Y.append(start_y)
    return Y

def meth_epsilon(y0, t0, tf, eps, f, meth):
    start_y = y0
    start_x = t0
    h = (tf - t0) * eps
    Y = np.array([start_y])
    while (start_x < tf):
        start_y = meth(start_y, start_x, h, f)
        start_x += h
        Y = np.append(Y, start_y)
    return Y

#==============TANGENTS===============#

def tangents_field(y0, t0, h, f, meth, N):
    t0_ = t0
    X = np.empty(N)
    Y = np.empty(N)
    DY = np.empty(N)
    X[0] = t0
    Y[0] = y0
    DY[0] = f(y0, t0)
    for i in range(1,N):
        y0 = meth(y0, t0, h, f)
        t0 += h
        X[i] = t0
        Y[i] = y0
        DY[i] = f(y0, t0)
    return X, Y, DY

#==============TEST_ZONE==============#

def test_methodes():
    # Dimension 1
    t0 = 0.
    y0 = 1.
    N = 10000
    h = 0.01
    f = lambda y,t : y/(1 + t**2)

    #print(meth_n_step(y0, t0, N, h, f, step_euler))
    #print(meth_n_step(y0, t0, N, h, f, step_middle_point))
    #print(meth_n_step(y0, t0, N, h, f, step_heun))
    #print(meth_n_step(y0, t0, N, h, f, step_runge_kutta_4))

    # Plot Section

    X = np.arange(t0, t0 + (N)*h, h)
    Y_euler = meth_n_step(y0, t0, N, h, f, step_euler)
    Y_middle_point = meth_n_step(y0, t0, N, h, f, step_middle_point)
    Y_heun = meth_n_step(y0, t0, N, h, f, step_heun)
    Y_runge_kutta_4 = meth_n_step(y0, t0, N, h, f, step_runge_kutta_4)

        # Different Methods Comparison

    y_sol = lambda t : np.exp(np.arctan(t))

    plt.plot(X, Y_euler, label='Euler')
    plt.plot(X, Y_middle_point, label='Middle point')
    plt.plot(X, Y_heun, label='Heun')
    plt.plot(X, Y_runge_kutta_4, label='Runge-Kutta 4')
    plt.plot(X, y_sol(X), label='solution')
    plt.title('n_step : solution to y\'(t) = y(t)/(1 + t^2) with y(0) = 1')
    plt.legend()
    plt.show()

    plt.plot(X, Y_euler - y_sol(X), label='Euler')
    plt.plot(X, Y_middle_point - y_sol(X), label='Middle point')
    plt.plot(X, Y_heun - y_sol(X), label='Heun')
    plt.plot(X, Y_runge_kutta_4 - y_sol(X), label='Runge-Kutta 4')
    plt.title('distance to solution to y\'(t) = y(t)/(1 + t^2) with y(0) = 1')
    plt.legend()
    plt.show()

    # Dimension 2

    g = lambda Y,t : np.array([-Y[1],  Y[0]])
    Y0 = np.array([1, 0])
    g_sol_1 = lambda t : np.cos(t-np.pi/2)
    g_sol_2 = lambda t : -np.sin(t-np.pi/2)

    # Plot Section

    X = np.arange(t0, t0 + (N)*h, h)
    Y_euler = meth_n_step(Y0, t0, N, h, g, step_euler)
    Y_middle_point = meth_n_step(Y0, t0, N, h, g, step_middle_point)
    Y_heun = meth_n_step(Y0, t0, N, h, g, step_heun)
    Y_runge_kutta_4 = meth_n_step(Y0, t0, N, h, g, step_runge_kutta_4)

        # Different Methods Comparison

    #y_sol = lambda t : np.exp(np.arctan(t))

    plt.plot(X, Y_euler, label='Euler')
    plt.plot(X, Y_middle_point, label='Middle point')
    plt.plot(X, Y_heun, label='Heun')
    plt.plot(X, Y_runge_kutta_4, label='Runge-Kutta 4')
    plt.plot(X, g_sol_1(X), label='solution y1')
    plt.plot(X, g_sol_2(X), label='solution y2')
    plt.title('n_step : solution to y\'(t) = [-y2(t), y1(t)] with y(0) = [1, 0]')
    plt.legend()
    plt.show()

    # Tangents Field

    # Dimension 1

        # Theoretical
    X = np.arange(0.,4.,0.2)
    U, V = np.meshgrid(X, X)
    dy_sol = lambda t : sm.derivative(y_sol, t)
    plt.quiver(U, V, y_sol(X), dy_sol(X))
    plt.title("Theoretical Tangents Field")
    plt.show()

        # Experimental
    N = 20
    h = 0.2

    X, Y, DY = tangents_field(y0, t0, h, f, step_euler, N)
    U, V = np.meshgrid(X, X)
    plt.quiver(U, V, Y, DY)
    plt.title("Experimental Tangents Field")
    plt.show()

    # Dimension 2
    t0 = 0.
    y0 = np.array([1, 0])
    N = 100
    h = 0.01
    f = lambda y,t : np.array([-y[1], y[0]])

    return

if __name__ == '__main__':
    test_methodes()
