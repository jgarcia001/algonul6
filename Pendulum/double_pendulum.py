import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import methodes as m

#---------------------------------------------------------------------
#Constantes utilisees lors des tests

g = 9.81
N = 500
h = 0.05

w = 1.0
l = 1.0

t1 =-3
t2 =3

#---------------------------------------------------------------------
#Fonctions implementees pour le pendule a deux maillons

#Retourne la position (x,y) du premier solide
def pos1(l_1, theta):
    x1 = l_1 * np.sin(theta)
    y1 = -l_1 * np.cos(theta)
    return x1, y1

#Retourne la position (x,y) du deuxième solide
def pos2(l_1, l_2, theta1, theta2):
    x1, y1 = pos1(l_1, theta1)
    x2 = x1 + l_2 * np.sin(theta2)
    y2 = y1 - l_2 * np.cos(theta2)
    return x2, y2

#Retourne sous la forme d'une lambda-expression un tableau contenant:
# -les vitesses des solides
# -les accélérations des solides
def f_function():

    return (lambda u,t: np.array([u[1],

                                  (- g *(3*np.sin(u[0]) + np.sin(u[0] - 2*u[2]))
                                   - 2*np.sin(u[0] - u[2])*((u[3]**2)*l + (u[1]**2)*l * np.cos(u[0] - u[2])))
                                  / (l *(3 - np.cos(2*(u[0] - u[2])))),

                                  u[3],

                                  (2*np.sin(u[0] - u[2])*((u[1]**2)*l*2
                                                          + g*2*np.cos(u[0])
                                                          + (u[3]**2)*l*np.cos(u[0] - u[2])))
                                  / (l*(3 - np.cos(2*(u[0] - u[2]))))]))
#Retourne les solutions de l'équation du mouvement en fonction
# des conditions initiales (y_zero)
def double_pendulum_position_function(y_zero):

    f = f_function()
    return m.meth_n_step(y_zero, 0, N, h, f, m.step_runge_kutta_4)

#---------------------------------------------------------------------
#Tests realises pour tester les fonctions precedentes

#Retourne les solutions de l'équation du mouvement avec :
# -un tableau pour l'angle theta1
# -un tableau pour l'angle theta2
# -le temps du premier retournement
def double_pendulum(t1, t2):
    u_init = np.array([t1, 0, t2, 0])
    tab = double_pendulum_position_function(u_init)
    first_flip = 1
    first_flip_b = False

    the1 = np.empty(N)
    the2 = np.empty(N)
    for i in range (1, N):

        the1[i] = tab[i][0]

        if ( tab[i][2] > np.pi or tab[i][2] < -np.pi):
            the2[i] = (tab[i][2] % 2*np.pi) - np.pi
            if(first_flip_b == False):
                first_flip = N / i
                first_flip_b = True
        else :
            the2[i] = tab[i][2]

    return the1, the2, first_flip


#Affiche le grahique de l'angle du deuxième solide en fonction
#de celui du premier solide.
def test_modelling():
    the1, the2, first_flip = double_pendulum(t1, t2)
    plt.plot(the1, the2)
    plt.show()

test_modelling()

#Affiche le graphique de la position du deuxième pendule lors
# de la simulation: y en fonction de x
def test_modelling_position():
    the1, the2, firstflip = double_pendulum(t1, t2)

    size = np.size(the1)

    x_pos = np.zeros(size)
    y_pos = np.zeros(size)

    for j in range(size):
        x,y = pos2(l, l, the1[j], the2[j])
        x_pos[j] = x
        y_pos[j] = y

    plt.plot(x_pos, y_pos)
    plt.show()

test_modelling_position()

#Affiche la carte des premiers retournements suivant les conditions in\
itiales
def test_flip():
    N = 200
    h = 0.05

    x_start = -3
    x_end = 3

    y_start = -3
    y_end = 3

    interval = 200
    space =  (x_end - x_start)/interval

    image = np.empty((interval, interval))

    xaxis = np.arange(x_start, x_end, space)
    yaxis = np.arange(y_start, y_end, space)
    for i in range(0, interval,1):
        for j in range(0, interval,1):
            if(3*np.cos(xaxis[i]) + np.cos(yaxis[j]) <= 2):
                the1, the2, first_flip = double_pendulum(xaxis[i], yaxis[j])
            else:
                first_flip = 0
            image[j][i] = first_flip

    plt.imshow(image, cmap= plt.get_cmap('Paired'))
    plt.show()

test_flip()
