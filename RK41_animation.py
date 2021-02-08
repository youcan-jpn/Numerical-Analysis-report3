import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
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
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.set_xlabel('x /m')
ax1.set_ylabel('y /m')
ax2.set_xlabel('time /s')
ax2.set_ylabel('energy /J')
ax3.set_xlabel('time /s')
ax3.set_ylabel('kinetic energy /J')
ax4.set_xlabel('time /s')
ax4.set_ylabel('potential energy /J')


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

images = []
ax2.plot(t_series, H_series, c="red", label="H")
ax3.plot(t_series, T_series, c="green", label="T")
ax4.plot(t_series, U_series, c="blue", label="U")

for i in range(steps):
    x = [0, xs[i]]
    y = [0, ys[i]]
    # T = T_series[i]
    # U = U_series[i]
    # H = T + U
    image = ax1.plot(x, y, 'o-', lw=2, c="black", label="pendulum")
    ax1.grid(True)
    ax1.axis('equal')
    images.append(image)

ani = anim.ArtistAnimation(fig, images, interval=10)
ax2.legend(loc="upper right")
ax3.legend(loc="upper right")
ax4.legend(loc="upper right")
plt.tight_layout()

# ani.save('./figure/RK41/animation_{0}-{1}-{2}-{3}{4}{5}.gif'.format(
#     dt_now.year, dt_now.month, dt_now.day,
#     dt_now.hour, dt_now.minute, dt_now.second), writer='pillow', fps=50)
plt.show()
