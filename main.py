import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from collections import deque

# constants
g = 9.80665  # Standard gravity

# parameters
l1 = 1
m1 = 1
theta1 = np.pi/2
w1 = 0
t_start = 0
t_end = 10
steps = 1000


dt = (t_end - t_start)/steps

# lists
theta_series = deque([theta1])
w_series = deque([w1])
xs = deque([l1*np.sin(theta1)])
ys = deque([-l1*np.cos(theta1)])
H_series = deque([m1*(l1**2)*(w1**2)/2 - m1*g*l1*np.cos(theta1)])

fig, ax = plt.subplots()
plt.xlabel('x')
plt.ylabel('y')


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
    H = m1*(l1**2)*(w**2)/2 - m1*g*l1*np.cos(theta)
    H_series.append(H)


for i in range(steps):
    theta = theta_series[-1]
    w = w_series[-1]
    RungeKutta41(theta, w)

images = []
for i in range(steps+1):
    x = [0, xs[i]]
    y = [0, ys[i]]
    image = plt.plot(x, y, 'o-', lw=2, c="black")
    ax.grid(True)
    ax.axis('equal')
    images.append(image)

ani = anim.ArtistAnimation(fig, images, interval=10)
plt.show()
print(H_series)
