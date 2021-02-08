import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from collections import deque
from datetime import datetime

# constants
g = 9.80665  # Standard gravity

# parameters
l1 = 1.0
l2 = 1.0
m1 = 1.0
m2 = 1.0
theta10 = np.pi/6
theta20 = np.pi/6
w10 = 0.0
w20 = 0.0
t_start = 0
t_end = 20
steps = 2000

# calculated
dt = (t_end - t_start)/steps

# lists
theta1_series = deque([theta10])
theta2_series = deque([theta20])
w1_series = deque([w10])
w2_series = deque([w20])
x1s = deque([l1*np.sin(theta10)])
y1s = deque([-l1*np.cos(theta10)])
x2s = deque([l1*np.sin(theta10)+l2*np.sin(theta20)])
y2s = deque([-l1*np.cos(theta10)-l2*np.cos(theta20)])
t_series = np.linspace(t_start, t_end, steps)
T1_series = deque([m1*(l1**2)*(w10**2)/2])
T2_series = deque([m2*((l1**2)*(w10**2)+(l2**2)*(w20**2)
                  + 2*l1*l2*np.cos(theta10-theta20)*w10*w20)])
U1_series = deque([-m1*g*l1*np.cos(theta10)])
U2_series = deque([-m2*g*(l1*np.cos(theta10)+l2*np.cos(theta20))])
H_series = deque([T1_series[0]+T2_series[0]+U1_series[0]+U2_series[0]])

fig, ax = plt.subplots()
ax.set_xlabel('x /m', fontsize="14")
ax.set_ylabel('y /m', fontsize="14")
ax.grid()
fig.suptitle("$m_1={0}, m_2={1}, l_1={2}, l_2={3}$".format(m1, m2, l1, l2),
             fontsize="20")


def f(theta1, theta2, w1, w2):
    phi = theta1 - theta2
    numerator = (-m2*l2*(w2**2)*np.sin(phi)
                 - (m1+m2)*g*np.sin(theta1)
                 - m2*l1*(w1**2)*np.sin(phi)*np.cos(phi)
                 + m2*g*np.cos(phi)*np.sin(theta2))
    denominator = (m1+m2)*l1-m2*l1*((np.cos(phi))**2)
    value = numerator/denominator
    return value


def h(theta1, theta2, w1, w2):
    phi = theta1-theta2
    numerator = (m2*l2*(w2**2)*np.cos(phi)*np.sin(phi)
                 + (m1+m2)*g*np.sin(theta1)*np.cos(phi)
                 + (m1+m2)*l1*(w1**2)*np.sin(phi)
                 - (m1+m2)*g*np.sin(theta2))
    denominator = (m1+m2)*l2 - m2*l2*((np.cos(phi)**2))
    value = numerator/denominator
    return value


def RungeKutta42(theta1, theta2, w1, w2, dt):
    k11 = f(theta1, theta2, w1, w2)  # w1
    n11 = w1  # theta1
    k21 = h(theta1, theta2, w1, w2)  # w2
    n21 = w2  # theta2

    k12 = f(theta1+n11*dt/2, theta2+n21*dt/2, w1+k11*dt/2, w2+k21*dt/2)
    n12 = w1 + k11*dt/2
    k22 = h(theta1+n11*dt/2, theta2+n21*dt/2, w1+k11*dt/2, w2+k21*dt/2)
    n22 = w2 + k21*dt/2

    k13 = f(theta1+n12*dt/2, theta2+n22*dt/2, w1+k12*dt/2, w2+k22*dt/2)
    n13 = w1 + k12*dt/2
    k23 = h(theta1+n12*dt/2, theta2+n22*dt/2, w1+k12*dt/2, w2+k22*dt/2)
    n23 = w2 + k22*dt/2

    k14 = f(theta1+n13*dt, theta2+n23*dt, w1+k13*dt, w2+k23*dt)
    n14 = w1 + k13*dt
    k24 = h(theta1+n13*dt, theta2+n23*dt, w1+k13*dt, w2+k23*dt)
    n24 = w2 + k23*dt

    w1 = w1 + (k11/6 + k12/3 + k13/3 + k14/6)*dt
    theta1 = theta1 + (n11/6 + n12/3 + n13/3 + n14/6)*dt

    w2 = w2 + (k21/6 + k22/3 + k23/3 + k24/6)*dt
    theta2 = theta2 + (n21/6 + n22/3 + n23/3 + n24/6)*dt

    w1_series.append(w1)
    w2_series.append(w2)

    theta1_series.append(theta1)
    theta2_series.append(theta2)

    x1s.append(l1*np.sin(theta1))
    y1s.append(-l1*np.cos(theta1))

    x2s.append(l1*np.sin(theta1)+l2*np.sin(theta2))
    y2s.append(-l1*np.cos(theta1)-l2*np.cos(theta2))

    phi = theta1-theta2

    T1 = m1*(l1**2)*(w1**2)/2
    T2 = m2*((l1**2)*(w1**2)+(l2**2)*(w2**2)
             + 2*l1*l2*np.cos(phi)*w1*w2)
    T1_series.append(T1)
    T2_series.append(T2)

    U1 = -m1*g*l1*np.cos(theta1)
    U2 = -m2*g*(l1*np.cos(theta1)+l2*np.cos(theta2))
    U1_series.append(U1)
    U2_series.append(U2)

    H = T1 + T2 + U1 + U2
    H_series.append(H)


for i in range(steps):
    theta1 = theta1_series[-1]
    theta2 = theta2_series[-1]
    w1 = w1_series[-1]
    w2 = w2_series[-1]
    RungeKutta42(theta1, theta2, w1, w2, dt)

images = []

for i in range(steps):
    x = [0, x1s[i], x2s[i]]
    y = [0, y1s[i], y2s[i]]

    image = ax.plot(x, y, 'o-', lw=2, c="black")
    ax.grid(True)
    ax.axis('equal')
    images.append(image)

ani = anim.ArtistAnimation(fig, images, interval=10)


dt_now = datetime.now()

ani.save('./figure/temp/animation_{0}-{1}-{2}-{3}{4}{5}.gif'.format(
    dt_now.year, dt_now.month, dt_now.day,
    dt_now.hour, dt_now.minute, dt_now.second), writer='pillow', fps=50)
# plt.show()
