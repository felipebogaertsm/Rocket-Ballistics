import pandas as pd
import numpy as np
import fluids.atmosphere as atm
import matplotlib.pyplot as plt

from functions import *

# ______________________________________________________________
# MOTOR DATA READ FROM CSV (SRM SOLVER PROGRAM)

motordata = pd.read_csv('motordata.csv')
t_file = np.array(motordata.Time)
T_file = np.array(motordata.Thrust)
mp_file = np.array(motordata.Prop_mass)

# ______________________________________________________________
# INPUTS

# Initial conditions [m, m/s, s]:
y, v, t = np.array([0]), np.array([0]), np.array([0])
# Initial height above sea level [m]:
h0 = 4
# Launch rail length [m]:
rain_length = 5
# Time step (0.01 recommended) [s]:
dt = 0.01
# Drag coefficient:
Cd = 0.39 * 1.1
# Rocket mass (without motor) and payload mass [kg]:
m_rocket, m_payload = 24, 4
# Empty motor mass [kg]:
m_motor = 20
# Rocket radius [m]:
r = 0.5 * 180e-3

# Parachute data:
# Time after apogee for drogue parachute activation [s]:
drogue_time = 1
# Drogue drag coefficient:
Cd_drogue = 1.75
# Drogue effective diameter [m]:
D_drogue = .3
# Main parachute drag coefficient [m]:
Cd_main = 1.75
# Main parachute effective diameter [m]:
D_main = 0.8
# Main parachute height activation [m]:
main_chute_activation_height = 450

# ______________________________________________________________
# BALLISTICS SIMULATION

mp = np.array(mp_file[0])
apogee = 0
apogee_time = -1
main_time = 0

i = 0
while y[i] >= 0:

    T = np.interp(t, t_file, T_file, left=0, right=0)
    mp = np.interp(t, t_file, mp_file, left=mp_file[0], right=0)

    if i == 0:
        a = np.array([T[0] * (m_rocket + m_payload + mp[0] + m_motor) * 0])

    # Local density 'p_air' [kg/m3] and acceleration of gravity 'g' [m/s2].
    p_air = atm.ATMOSPHERE_1976(y[i] + h0).rho
    g = atm.ATMOSPHERE_1976.gravity(h0 + y[i])

    # Instantaneous mass of the vehicle [kg]:
    M = m_rocket + m_payload + m_motor + mp[i]

    if i == 0:
        Minitial = M

    # Drag properties:
    if v[i] < 0 and y[i] <= main_chute_activation_height and mp[i] == 0:
        if main_time == 0:
            main_time = t[i]
        Adrag = (np.pi * r ** 2) * Cd + (np.pi * D_drogue ** 2) * 0.25 * Cd_drogue + \
                (np.pi * D_main ** 2) * 0.25 * Cd_main
    elif apogee_time >= 0 and t[i] >= apogee_time + drogue_time:
        Adrag = (np.pi * r ** 2) * Cd + (np.pi * D_drogue ** 2) * 0.25 * Cd_drogue
    else:
        Adrag = (np.pi * r ** 2) * Cd

    D = (Adrag * p_air) * 0.5

    k1, l1 = ballisticsODE(y[i], v[i], T[i], D, M, g)
    k2, l2 = ballisticsODE(y[i] + 0.5 * k1 * dt, v[i] + 0.5 * l1 * dt, T[i], D, M, g)
    k3, l3 = ballisticsODE(y[i] + 0.5 * k2 * dt, v[i] + 0.5 * l2 * dt, T[i], D, M, g)
    k4, l4 = ballisticsODE(y[i] + 0.5 * k3 * dt, v[i] + 0.5 * l3 * dt, T[i], D, M, g)

    y = np.append(y, y[i] + (1 / 6) * (k1 + 2 * (k2 + k3) + k4) * dt)
    v = np.append(v, v[i] + (1 / 6) * (l1 + 2 * (l2 + l3) + l4) * dt)
    a = np.append(a, (1 / 6) * (l1 + 2 * (l2 + l3) + l4))
    t = np.append(t, t[i] + dt)

    if y[i + 1] <= y[i] and mp[i] == 0 and apogee == 0:
        apogee = y[i]
        apogee_time = t[np.where(y == apogee)]

    i = i + 1

if y[-1] < 0:
    y = np.delete(y, -1)
    v = np.delete(v, -1)
    a = np.delete(a, -1)
    t = np.delete(t, -1)

vRail = v[np.where(y >= rain_length)]
vRail = vRail[0]
yburnout = y[np.where(v == np.max(v))]

# ______________________________________________________________
# RESULTS

print('\n')
print('Apogee: %.1f meters' % apogee)
print('Maximum velocity: %.1f m/s or Mach %.4f' % (np.max(v),
                        (np.max(v) / atm.ATMOSPHERE_1976(y[np.where(v == np.max(v))]).v_sonic)))
print('Maximum acceleration: %.1f m/s-s or %.2f gs' % (np.max(a), np.max(a)/g))
print('Initial mass of the vehicle: %.3f kg' % (Minitial))
print('Time to apogee %.1f seconds' % (t[np.where(y == apogee)]))
print('Time to reach ground: %.1f seconds' % (t[-1]))
print('Velocity out of the launch rail: %.1f m/s' % (vRail))
print('Drogue terminal velocity: %.1f m/s' % np.abs(v[np.where(t == main_time - dt)]))
print('Main chute terminal velocity: %.1f m/s' % np.abs(v[-1]))

# ______________________________________________________________
# PLOTS

fig1 = plt.figure()

plt.subplot(3, 1, 1)
plt.ylabel('Height (m)')
plt.grid(linestyle='-.')
plt.plot(t, y, color='b')
plt.subplot(3, 1, 2)
plt.ylabel('Velocity (m/s)')
plt.grid(linestyle='-.')
plt.plot(t, v, color='g')
plt.subplot(3, 1, 3)
plt.ylabel('Acc (m/s2)')
plt.xlabel('Time (s)')
plt.grid(linestyle='-.')
plt.plot(t, a, color='r')
plt.show()
