from OrderStateMachine_R import *
import sol_R as ground
import avec_ressort_sans_sol_R as air
import numpy as np
import pygame as pg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import imageio
import os


#paramètres de simulation globaux

dt = 5e-3
n = 5
W, H = 700, 400
hauteur = 0.4
T=0.5
dx3=0.5
K=0.2
theta_moy=0
boo=True
C3=45

#initialisation 

q0 = np.array([0, 0, 1, 0, 0.07])
dq0 = np.zeros(n)
y = np.concatenate((q0, dq0))
state_machine = OrderStateMachine()

def kinetic_energy_without_M(q, dq, params):
    theta1, theta2, w, x0, y0 = q
    dtheta1, dtheta2, dw, dx0, dy0 = dq

    # Paramètres
    m1 = params["m1"]
    m3 = params["m3"]
    r1 = params["r1"]
    r2 = params["r2"]
    I1 = params["I1"]
    I3 = params["I3"]

    # Vitesse du centre de masse du bras 1 (rotation autour de x0,y0)
    v1x = -r1 * np.sin(theta1) * dtheta1
    v1y =  r1 * np.cos(theta1) * dtheta1
    v1_sq = v1x*2 + v1y*2

    # Vitesse du centre de masse du bras 3 (à distance w + r2 du pivot)
    vxA = w * np.sin(theta1) * dtheta1 + np.cos(theta1) * dw
    vyA = -w * np.cos(theta1) * dtheta1 + np.sin(theta1) * dw

    v3x = vxA - r2 * np.sin(theta1 + theta2) * (dtheta1 + dtheta2)
    v3y = vyA + r2 * np.cos(theta1 + theta2) * (dtheta1 + dtheta2)
    v3_sq = v3x*2 + v3y*2

    # Énergie cinétique
    Ec = 0.5 * I1 * dtheta1*2 + 0.5 * I3 * dtheta2*2 + 0.5 * m1 * v1_sq + 0.5 * m3 * v3_sq
    return Ec

def calculate_energy(q, dq, params):
    theta1, theta2, w, x0, y0 = q
    dtheta1, dtheta2, dw, dx0, dy0 = dq
    
    # Énergie cinétique
    kinetic = kinetic_energy_without_M(q,dq,params)
    
    # Énergie potentielle gravitationnelle
    m1, m3 = params["m1"], params["m3"]
    r1, r2 = params["r1"], params["r2"]
    g = params["g"]
    
    # Position des centres de masse
    y1 = y0 + r1 * np.cos(theta1)  # Segment 1
    y2 = y0 + w * np.cos(theta1) + r2 * np.cos(theta2)  # Segment 2
    
    potential_gravity = m1 * g * y1 + m3 * g * y2
    
    # Énergie potentielle élastique
    k = params["k"]
    l0 = params["l0"]
    X = params["X"]
    potential_spring = 0.5 * k * (w - l0 + X)**2
    
    # Énergie potentielle du sol (si en contact)
    k_sol = params["k_sol"]
    potential_ground = 0.5 * k_sol * min(y0, 0)**2
    
    total_energy = kinetic + potential_gravity + potential_spring + potential_ground
    return total_energy

def new_X(q, dq, hauteur, params):
    k = params["k"]
    m1, m3 = params["m1"], params["m3"]
    r1, r2 = params["r1"], params["r2"]
    g = params["g"]
    l0, X = params["l0"], params["X"]
    w = q[2]

    A = -w + l0 + X
    E_target = m1*g*(hauteur + r1) + m3*g*(hauteur + r2 + l0)
    E_current = calculate_energy(q, dq, params)
    delta_E = E_target - E_current

    inside_sqrt = A**2 + 2*delta_E/k

    if inside_sqrt < 0:
        return -A # revient à X = 0
    elif delta_E >= 0:
        return (np.sqrt(inside_sqrt) - A)
    else:
        return -(np.sqrt(inside_sqrt) - A)

    
    
def detection_event(y, params):
    q = y[:n]
    dq = y[n:]
    y0 = q[4]
    X = params.get("X", 0)
    l0 = params.get("l0", 0)
    w = q[2]
    
    if y0 > 0:
        return "y_0 > 0"
    elif y0 <= 0 and state_machine.current_state == "aloft":
        return "y_0 <= 0"
    elif dq[2] >= 0 and state_machine.current_state == "spring_compression":
        return "X >= X_d"
    return None

dq2_prev = 0
epsilon = 1e-5
correction_done = False

def detection_sol(y):
    global dq2_prev, correction_done
    q = y[:n]
    dq = y[n:]
    y0 = q[4]
    dy0 = dq[4]
    dq2 = dq[2]

    # 1. À l'atterrissage, on mémorise la vitesse de compression
    if y0 <= 0 and dy0 < 0:
        dq2_prev = dq2

    # 2. Quand on atteint la compression maximale (dq2 change de signe)
    if y0 <= 0 and dq2_prev < -epsilon and dq2 >= epsilon:
        delta = new_X(q, dq, hauteur, ground.params)
        ground.params["X"] += delta
        print("Correction X appliquée :", delta)
    dq2_prev = dq2

    # 3. Quand le robot décolle (retourne dans l'air), reset X
    if y0 > 0 and dy0 > 0:
        ground.params["X"] = 0

    

def f(t,y):
    global boo
    global theta_moy
    global theta_1d
    global theta2_d
    q = y[:n]
    dq =y[n:]
    event = detection_event(y, ground.params)
    if event:
        state_machine.transition(event)
        detection_sol(y)
        
        
    state = state_machine.current_state
    
    if state == "aloft":
        boo=True
        dx=(dq[3]+dq[2]*np.sin(q[0])+q[2]*dq[0]*np.cos(q[0]))
        delta_x=((dx3*T)/2)+K*(dx-dx3)
        theta_1d=-np.arcsin(delta_x/q[2])
        M_val = air.M_matrix(q,air.params)
        C_val = air.C_matrix(q, dq,air.params)
        G_val = air.G_vector(q,dq,air.params)
        Gamma_val = air.control_forces(t, q, dq)
        Gamma_val[0]=-2000*(q[0]-theta_1d)+10*dq[0]
    else:
        if boo==True:
            theta2_d=theta_moy
            theta_moy=0
            boo=False
        else:
            boo=False
        theta_moy+=q[1]*dt/T
        M_val = ground.M_matrix(q,ground.params)
        C_val = ground.C_matrix(q, dq,ground.params)
        G_val = ground.G_vector(q,dq,ground.params)
        Gamma_val = ground.control_forces(t, q, dq)
        Gamma_val[1]=-2300*(q[1]-theta2_d)-500*dq[1]
        Gamma_val[3]=-C3*dq[3]
    ddq = np.linalg.solve(M_val, Gamma_val - C_val @ dq - G_val)# Résolution de M(q) * ddq = Γ - C(q, dq) * dq - G(q)
    return np.concatenate((dq, ddq)) 

def update(y): 
    sol = solve_ivp(f, [0., dt], y, method="RK45")
    new_y = sol.y[:, -1]
    return new_y

def draw(y, screen):
    q = y[:n]   # Positions généralisées
    dq = y[n:]
    theta1, theta2, w, x0, y0 = q
    r1 = ground.params["r1"]
    r2 = ground.params["r2"]
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


def run(y, save_video=True, video_filename="simulation_hopper.mp4"):
    pg.init()
    screen = pg.display.set_mode([W, H])
    running = True
    frame_count = 0
    frames = []
    
    # Initialisation des listes pour les courbes
    w_list = []
    yB_list = []
    y0_list = []
    t_list = []
    t = 0

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        if frame_count >= 1/10:
            screen.fill((30, 30, 30))
            draw(y, screen)
            pg.display.flip()
            
            if save_video:
                # Capture de la frame sous forme de tableau numpy
                frame = pg.surfarray.array3d(pg.display.get_surface())
                # Conversion de (W, H, 3) -> (H, W, 3)
                frame = np.transpose(frame, (1, 0, 2))
                frames.append(frame)
            
            frame_count = 0
            pg.time.delay(33)
        frame_count += dt
        
        q = y[:n]
        dq = y[n:]

        
        y = update(y)

        t += dt
        
        # Stockage des données
        
        r2 = ground.params["r2"]
        yA=q[4]+q[2]*np.cos(q[0])
        yB=yA+r2*np.cos(q[1])
        w_list.append(q[2])
        yB_list.append(yB)
        y0_list.append(q[4])
        t_list.append(t)
        
        print(calculate_energy(q, dq, ground.params))

    pg.quit()
    
     # Affichage des courbes
    plt.figure(figsize=(10, 4))
    plt.plot(t_list, w_list, label="w = q[2]")
    plt.plot(t_list, yB_list, label="yB ")
    plt.plot(t_list, y0_list, label="y0 ")
    plt.xlabel("Temps (s)")
    plt.ylabel("Valeurs")
    plt.title("Évolution de w et y0 au cours du temps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # if save_video and frames:
    #     print("Enregistrement de la vidéo...")
    #     with imageio.get_writer(video_filename, fps=int(1/dt), format='ffmpeg') as writer:
    #         for frame in frames:
    #             writer.append_data(frame)
    # print(f"Vidéo enregistrée : {video_filename}")


run(y)
