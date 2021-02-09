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
theta1 = np.pi/4
w1 = 0
t_start = 0
t_end = 10
steps = 1000

# calculated
dt = (t_end - t_start)/steps

# lists
theta_series = deque([theta1])
w_series = deque([w1])
xs = deque([l1*np.sin(theta1)])
ys = deque([-l1*np.cos(theta1)])
t_series = np.linspace(t_start, t_end, steps+1)
T_series = deque([m1*(l1**2)*(w1**2)/2])
U_series = deque([-m1*g*l1*np.cos(theta1)])
H_series = deque([m1*(l1**2)*(w1**2)/2-m1*g*l1*np.cos(theta1)])

fig = plt.figure()
ax = fig.add_subplot(111)


def f(theta):
    return -g*np.sin(theta)/l1


def RungeKutta41(theta, w):
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
    xs.append(l1*np.sin(theta))
    ys.append(-l1*np.cos(theta))
    T = m1*(l1**2)*(w**2)/2
    U = - m1*g*l1*np.cos(theta)
    H = T + U
    T_series.append(T)
    U_series.append(U)
    H_series.append(H)


for i in range(steps):
    theta = theta_series[-1]
    w = w_series[-1]
    RungeKutta41(theta, w)


def guide(t):
    return 3*10**(-9)*t


H_series = np.array(H_series)
H_exact = H_series[0]
ax.plot(t_series, H_exact-H_series, label="$E_0 - E$")
ax.plot(t_series, guide(t_series),
        label="guide:"+R"$f(t)=3*10^{-9}t$", ls="dashed")
ax.set_xlabel("time /s", fontsize=14)
ax.grid()
ax.legend()

plt.tight_layout()
fig.savefig('./figure/RK41/evaluation2/{0}-{1}-{2}-{3}{4}{5}.jpeg'
            .format(dt_now.year, dt_now.month, dt_now.day,
                    dt_now.hour, dt_now.minute, dt_now.second))
# plt.show()
