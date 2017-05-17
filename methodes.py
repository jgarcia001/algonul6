# Euler

def step_euler(y, t, h, f):
    return y + h * f(y,t)

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
    while (meth(start_y, start_x, h, f) - start_y > eps):
        start_y = meth(start_y, start_x, h, f)
        start_x += h
    return start_y
