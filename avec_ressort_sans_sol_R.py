import numpy as np
import pygame as pg
from scipy.integrate import solve_ivp

### SIMULATION PARAMETERS
n=5
dt = 5e-2       # timestep [s]
W = 700        # Width of screen [px] 
H = 400       # Height of screen [px]
k_2 = 10000
C_2 = 50
w_max = 1
params = {
    "m1": 1.0,     
    "m2": 0,   
    "m3": 10,     
    "r1": 0.5,    
    "r2": 0.4,    
    "I1": 1,    
    "I2": 0,  
    "I3": 10,
    "g":9.81,
    "k_sol":100000/3,
    "k":10000*1.1,
    "l0":1,
    "X":0
}
def M_matrix(q, params):
    theta1, theta2, w, x0, y0 = q
    m1, m3 = params["m1"], params["m3"]
    r1, r2 = params["r1"], params["r2"]
    I1, I3 = params["I1"], params["I3"]

    M = np.zeros((5, 5))
    M[0, 0] = I1 + m1 * r1*2 + m3 * w*2
    M[0, 1] = -m3 * r2 * w * np.cos(theta1 + theta2)
    M[0, 3] = (m1 * r1 + m3 * w) * np.cos(theta1)
    M[0, 4] = (-m1 * r1 - m3 * w) * np.sin(theta1)

    M[1, 0] = M[0, 1]
    M[1, 1] = I3 + m3 * r2**2
    M[1, 2] = -m3 * r2 * np.sin(theta1 + theta2)
    M[1, 3] = -m3 * r2 * np.cos(theta2)
    M[1, 4] = -m3 * r2 * np.sin(theta2)

    M[2, 1] = M[1, 2]
    M[2, 2] = m3
    M[2, 3] = m3 * np.sin(theta1)
    M[2, 4] = m3 * np.cos(theta1)

    M[3, 0] = M[0, 3]
    M[3, 1] = M[1, 3]
    M[3, 2] = M[2, 3]
    M[3, 3] = m1 + m3

    M[4, 0] = M[0, 4]
    M[4, 1] = M[1, 4]
    M[4, 2] = M[2, 4]
    M[4, 4] = m1 + m3

    return M
def C_matrix(q, dq, params):
    theta1, theta2, w, x0, y0 = q
    dtheta1, dtheta2, dw, dx0, dy0 = dq
    m1, m3 = params["m1"], params["m3"]
    r1, r2 = params["r1"], params["r2"]

    C = np.zeros((5, 5))

    C[0, 0] = 2 * m3 * w * dw
    C[0, 1] = 2 * m3 * r2 * w * np.sin(theta1 + theta2) * dtheta2
    C[0, 2] = 2 * m3 * w * dtheta1

    C[1, 0] = 2 * m3 * r2 * (w * np.sin(theta1 + theta2) * dtheta1 - np.cos(theta1 + theta2) * dw)
    C[1, 2] = -2 * m3 * r2 * np.cos(theta1 + theta2) * dtheta1

    C[2, 0] = -2 * m3 * w * dtheta1
    C[2, 1] = -2 * m3 * r2 * np.cos(theta1 + theta2) * dtheta2

    C[3, 0] = -2 * m1 * r1 * np.sin(theta1) * dtheta1 - 2 * m3 * (w * np.sin(theta1) * dtheta1 - np.cos(theta1) * dw)
    C[3, 1] = 2 * m3 * r2 * np.sin(theta2) * dtheta2
    C[3, 2] = 2 * m3 * np.cos(theta1) * dtheta1

    C[4, 0] = -2 * m1 * r1 * np.cos(theta1) * dtheta1 - 2 * m3 * (w * np.cos(theta1) * dtheta1 + np.sin(theta1) * dw)
    C[4, 1] = -2 * m3 * r2 * np.cos(theta2) * dtheta2
    C[4, 2] = -2 * m3 * np.sin(theta1) * dtheta1

    return C
def G_vector(q, dq, params):
    theta1, theta2, w, x0, y0 = q
    dtheta1, dtheta2, dw, dx0, dy0 = dq
    m1, m3 = params["m1"], params["m3"]
    r1, r2 = params["r1"], params["r2"]
    g = params.get("g", 9.81)
    k = params.get("k", 0.0)
    l0 = params.get("l0", 0.0)
    X = params.get("X", 0.0)  # must be passed if used

    G = np.zeros(5)

    G[0] = -g * m1 * r1 * np.sin(theta1) \
           - g * m3 * w * np.sin(theta1) \
           - m3 * r2 * w * np.sin(theta1 + theta2) * dtheta2**2 \
           - 2 * m3 * w * dtheta1 * dw

    G[1] = m3 * r2 * (-g * np.sin(theta2) \
                     - w * np.sin(theta1 + theta2) * dtheta1**2 \
                     + 2 * np.cos(theta1 + theta2) * dtheta1 * dw)

    if l0 + X - w > 0:
        G[2] = g * m3 * np.cos(theta1) - k * l0 - k * X + k * w \
               + m3 * r2 * np.cos(theta1 + theta2) * dtheta2**2 \
               + m3 * w * dtheta1**2
    else:
        K2 = params.get("K2", k)  # fallback
        G[2] = -K2 * l0 - K2 * X + K2 * w + g * m3 * np.cos(theta1) \
               + m3 * r2 * np.cos(theta1 + theta2) * dtheta2**2 \
               + m3 * w * dtheta1**2

    G[3] = m1 * r1 * np.sin(theta1) * dtheta1**2 \
           - m3 * r2 * np.sin(theta2) * dtheta2**2 \
           + m3 * w * np.sin(theta1) * dtheta1**2 \
           - 2 * m3 * np.cos(theta1) * dtheta1 * dw

    G[4] = g * (m1 + m3) \
           + m1 * r1 * np.cos(theta1) * dtheta1**2 \
           + m3 * r2 * np.cos(theta2) * dtheta2**2 \
           + m3 * w * np.cos(theta1) * dtheta1**2 \
           + 2 * m3 * np.sin(theta1) * dtheta1 * dw

    return G

def ind(a):
    if a >= w_max:
        return 1
    else:
        return 0
def control_forces(t, q, dq):
    """ Vecteur des forces externes Γ """
    gamma = np.zeros(n)
    gamma[2] = np.max(k_2*(q[2]-w_max), 0)
    return gamma

def f(t,y):
    q = y[:n]   # Positions généralisées (5)
    dq = y[n:]  # Vitesses généralisées (5)

    M_val = M_matrix(q,params)
    C_val = C_matrix(q, dq,params)
    G_val = G_vector(q,dq,params)
    Gamma_val = control_forces(t, q, dq)
# Résolution de M(q) * ddq = Γ - C(q, dq) * dq - G(q)
    ddq = np.linalg.solve(M_val, Gamma_val - C_val @ dq - G_val)
    return np.concatenate((dq, ddq)) 

def update(y):   
    sol = solve_ivp(f, [0., dt], y, method="RK45")
    new_y = sol.y[:, -1]
    return new_y


def draw(y, screen):
    q = y[:n]   # Positions généralisées
    dq = y[n:]
    theta1, theta2, w, x0, y0 = q
    r1 = params["r1"]
    r2 = params["r2"]
    scale=50
    x0d=x0*scale+(W//2)
    y0d=-y0*scale+(H//2)
    xA=x0d+w*np.sin(theta1)*scale
    xB=xA+r2*np.sin(theta2)*scale
    yA=y0d-w*np.cos(theta1)*scale
    yB=yA-r2*np.cos(theta2)*scale
    pg.draw.circle(screen, (210, 210, 210), (xB, yB), 8)
    pg.draw.line(screen, (210, 210, 210), (x0d, y0d ), (xA, yA), width=3)
    pg.draw.line(screen, (210, 210, 0), (xA, yA ), (xB, yB), width=3)
    pg.draw.line(screen, (210,0,210), (0,H//2), (W,H//2), width=3)

def run(y):
    pg.init()
    screen = pg.display.set_mode([W, H])
    running = True
    frame_count = 0

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        if frame_count >= 1/60:
            screen.fill((30, 30, 30))
            draw(y, screen)
            pg.display.flip()
            frame_count = 0
            pg.time.delay(33)
        frame_count += dt

        
        y = update(y)

    pg.quit()

if __name__ == "__main__":
    q0 = np.array([0, 0, 1, 0 ,0])# Positions initiales 
    dq0 = np.array([0, 0, 0, 0, 0])           # Vitesses initiales
    y0 = np.concatenate((q0, dq0))
    run(y0)
