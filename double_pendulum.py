import numpy as np
import matplotlib.pyplot as plt

import methodes as m

g = 9.81
w = 10

N = 100
h = 0.1 
l1 = 0.1
l2 = 0.2

u_init = np.array([np.pi/2, 0, np.pi/3, 0])

print(u_init)

def pos1(l, theta):
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    return x, y

def pos2(l1, theta1, l2, theta2):
    x1, y1 = pos1(l1, theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)


def f_function1():

    return (lambda u,t: np.array([u[1], (- 3*g*w* np.sin(u[0]) - w*g*np.sin(u[0] - 2*u[2]) - 2*np.sin(u[0] - u[2])*w*(u[3]**2*l2 + u[1]**2*l1 * np.cos(u[0] - u[2]))) / (l1*w(3 - np.cos(2*u[0] - 2*u[2]))), u[3], (2*np.sin(u[0] - u[2])*(u[1]**2*l1*2*w + g*2*w*np.cos(u[0]) + u[3]**2*w*np.cos(u[0] - u[2]))) / (l2*w(3 - np.cos(2*u[0] - 2*u[2])))]))


def f_function2():

    return (lambda u,t: np.array([u[3], (2*np.sin(u[0] - u[2])*(u[1]**2*l1*2*w + g*2*w*np.cos(u[0]) + u[3]**2*w*np.cos(u[0] - u[2]))) / (l2*w(3 - np.cos(2*u[0] - 2*u[2])))]))
    
def double_pendulum_position_function(y_zero):

    f = f_function1()
    return m.meth_n_step(y_zero, 0, N, h, f, m.step_runge_kutta_4)



print(double_pendulum_position_function(u_init))
