import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import methodes as m

#----------------------------------------------------------------------------

g = 9.81
N = 30
h = 0.5

w = 2.0
l1 = 1.0
l2 = 1.0

t1 =np.pi/2
t2 =np.pi/2

u_init = np.array([t1, 0, t2, 0])

#----------------------------------------------------------------------------

def pos1(l_1, theta):
    x1 = l_1 * np.sin(theta)
    y1 = -l_1 * np.cos(theta)
    return x1, y1

def pos2(l_1, l_2, theta1, theta2):
    x1, y1 = pos1(l_1, theta1)
    x2 = x1 + l_2 * np.sin(theta2)
    y2 = y1 - l_2 * np.cos(theta2)
    return x2, y2

def f_function():

    return (lambda u,t: np.array([u[1],

                                  (- g *(3*np.sin(u[0]) + np.sin(u[0] - 2*u[2]))
                                   - 2*np.sin(u[0] - u[2])*((u[3]**2)*l2 + (u[1]**2)*l1 * np.cos(u[0] - u[2])))
                                  / (l1*(3 - np.cos(2*(u[0] - u[2])))),

                                  u[3],

                                  (2*np.sin(u[0] - u[2])*((u[1]**2)*l1*2 
                                                          + g*2*np.cos(u[0]) 
                                                          + (u[3]**2)*l2*np.cos(u[0] - u[2])))
                                  / (l2*(3 - np.cos(2*(u[0] - u[2]))))]))
    
def double_pendulum_position_function(y_zero):

    f = f_function()
    return m.meth_n_step(y_zero, 0, N, h, f, m.step_runge_kutta_4)

#----------------------------------------------------------------------------

def double_pendulum():
    tab = double_pendulum_position_function(u_init)

    ligne, colonne = np.shape(tab)
    
    the1 = np.zeros(ligne)
    the2 = np.zeros(ligne)
    for i in range (ligne):
        the1[i] = tab[i][0]
        
        if ( tab[i][2] > np.pi):
            the2[i] = (tab[i][2] % 2*np.pi) - np.pi
        elif ( tab[i][2] < -np.pi):
            the2[i] = (tab[i][2] % 2*np.pi) - np.pi
        else :
            the2[i] = tab[i][2] 
        #print(the2[i])
    return the1, the2

#print( -np.sin( ((-5*np.pi/4 % np.pi) +np.pi)) +np.pi/4)
#print(np.sin(3*np.pi/4) +np.pi/4)

def test_modelling():
    the1, the2 = double_pendulum()
    #print(the2)
    plt.plot(the1, the2)
    plt.show()

#test_modelling()
    
def test_flip1():
    the1, the2 = double_pendulum()

    size = np.size(the1)
    
    ufx = np.zeros(size)
    ufy = np.zeros(size)
    
    fx = np.zeros(size)
    fy = np.zeros(size)
    po=0
    pox=0
    
    for p in range (size):
        
        if ((3*np.cos(the1[p]) + np.cos(the2[p])) >= 2):
            ufx[pox] = the1[p]
            ufy[pox] = the2[p]
            pox+=1
        else :
            fx[po] = the1[p]
            fy[po] = the2[p]
            po+=1
            
    plt.plot(fx, fy, "g+")
    plt.plot(ufx, ufy, "b+")
    plt.show()

#test_flip1()
    
def test_modelling_position():
    the1, the2 = double_pendulum()

    size = np.size(the1)
    
    x_pos = np.zeros(size)
    y_pos = np.zeros(size)
    
    for j in range(size):
        x,y = pos2(l1, l2, the1[j], the2[j])
        x_pos[j] = x
        y_pos[j] = y

    plt.plot(x_pos, y_pos)
    plt.show()

#test_modelling_position()
### TEST ###
the1, the2 = double_pendulum()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x_pos, y_pos = pos2(l1, l2, t1, t2)

def animate(i):
    x_pos, y_pos = pos2(l1, l2, the1[i], the2[i])
    #coord.set_xdata(np.append(coord.get_xdata, x_pos))
    #coord.set_ydata(np.append(coord.get_ydata, y_pos))
    ax.clear()
    ax.scatter(x_pos, y_pos)
    return coord,

anim = animation.FuncAnimation(fig, animate, frames = N, interval=h)

plt.show()
############
def test_flip2():
    unflip_a = np.empty(3600)
    unflip_b = np.empty(3600)
    
    flip_a = np.empty(3600)
    flip_b = np.empty(3600)
    '''
    white_a = np.empty(3600)
    white_b = np.empty(3600)
    '''
    g1=0
    g2=0
    
    a_debut = -3
    b_debut = -3
    
    while (a_debut <= 3):
        b_debut = -3
        while (b_debut <= 3):
            
            if (3*np.cos(a_debut) + np.cos(b_debut) >= 2):
                unflip_a[g1] = a_debut
                unflip_b[g1] = b_debut
                g1+=1
            else :
                flip_a[g2] = a_debut
                flip_b[g2] = b_debut
                g2 += 1
            b_debut += 0.1
        a_debut += 0.1

    plt.plot(flip_a, flip_b, "g+")
    plt.plot(unflip_a, unflip_b, "b+")
    plt.show()

#test_flip2()
