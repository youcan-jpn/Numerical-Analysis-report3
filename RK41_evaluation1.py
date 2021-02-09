import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime


# constants
g = 9.80665  # Standard gravity
dt_now = datetime.now()

# parameters
l1 = 1
m1 = 1
w0 = 0
t_start = 0
t_end = 10

# lists
T_series = deque([])
U_series = deque([])
H_series = deque([])


def f(theta):
    return -g*np.sin(theta)/l1


def RungeKutta41(theta, w, dt):
    k1 = f(theta)  # w
    n1 = w  # theta
    k2 = f(theta + n1*dt/2)
    n2 = w + k1*dt/2
    k3 = f(theta + n2*dt/2)
    n3 = w + k2*dt/2
    k4 = f(theta + n3*dt)
    n4 = w + k3*dt
    w = w + (k1/6 + k2/3 + k3/3 + k4/6)*dt
    theta = theta + (n1/6 + n2/3 + n3/3 + n4/6)*dt

    w_series.append(w)
    theta_series.append(theta)
    T = m1*(l1**2)*(w**2)/2
    U = - m1*g*l1*np.cos(theta)
    H = T + U
    T_series.append(T)
    U_series.append(U)
    H_series.append(H)


thetas = [np.pi/6, np.pi/4, np.pi/3, np.pi/2]
thetas_str = ["pi6", "pi4", "pi3", "pi2"]
thetas_tex = [R"\pi/6", R"\pi/4", R"\pi/3", R"\pi/2"]
dts = np.logspace(-4, 0, num=17, base=10.0)


def guide(t):
    return 10*t**4


for theta0, theta0_str, theta0_tex in zip(thetas, thetas_str, thetas_tex):
    fig, ax = plt.subplots()
    exact_H = m1*(l1**2)*(w0**2)/2 - m1*g*l1*np.cos(theta0)
    y = []
    for dt in dts:
        steps = int((t_end - t_start) / dt)
        theta_series = deque([theta0])
        w_series = deque([w0])
        for i in range(steps):
            theta = theta_series[-1]
            w = w_series[-1]
            RungeKutta41(theta, w, dt)
        y.append(np.abs(exact_H - H_series[-1]))
    ax.plot(dts, y, label=R"$\theta_0 ={}$".format(theta0_tex))
    ax.plot(dts, guide(dts), label=R"$10(\Delta t)^4$", ls="dashed")
    ax.semilogx()
    ax.semilogy()
    ax.grid()
    ax.legend(fontsize="14")
    ax.set_xlabel(R"$ \Delta t$", fontsize="14")
    ax.set_ylabel("Error", fontsize="14")
    fig.suptitle('Evaluation of RK4', fontsize="20")
    fig.savefig('./figure/RK41/evaluation1/{6}_{0}-{1}-{2}-{3}{4}{5}.jpeg'
                .format(dt_now.year, dt_now.month, dt_now.day, dt_now.hour,
                        dt_now.minute, dt_now.second, theta0_str))
