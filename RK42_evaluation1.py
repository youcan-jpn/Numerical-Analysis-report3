import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

# constants
g = 9.80665  # Standard gravity
dt_now = datetime.now()

# parameters
l1 = 1.0
l2 = 1.0
m1 = 1.0
m2 = 1.0
w10 = 0.0
w20 = 0.0
t_start = 0
t_end = 4


# lists
T1_series = deque([])
T2_series = deque([])
U1_series = deque([])
U2_series = deque([])
H_series = deque([])


def f(theta1, theta2, w1, w2):
    phi = (theta1 - theta2) % (2*np.pi)
    numerator = (-m2*l2*(w2**2)*np.sin(phi)
                 - (m1+m2)*g*np.sin(theta1)
                 - m2*l1*(w1**2)*np.sin(phi)*np.cos(phi)
                 + m2*g*np.cos(phi)*np.sin(theta2))
    denominator = (m1+m2)*l1-m2*l1*((np.cos(phi))**2)
    value = numerator/denominator
    return value


def h(theta1, theta2, w1, w2):
    phi = (theta1-theta2) % (2*np.pi)
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
    theta1 = (theta1 + (n11/6 + n12/3 + n13/3 + n14/6)*dt) % (2*np.pi)

    w2 = w2 + (k21/6 + k22/3 + k23/3 + k24/6)*dt
    theta2 = (theta2 + (n21/6 + n22/3 + n23/3 + n24/6)*dt) % (2*np.pi)

    w1_series.append(w1)
    w2_series.append(w2)

    theta1_series.append(theta1)
    theta2_series.append(theta2)

    phi = (theta1-theta2) % (2*np.pi)

    T1 = m1*(l1**2)*(w1**2)/2
    T2 = (m2*((l1**2)*(w1**2)+(l2**2)*(w2**2)
              + 2*l1*l2*np.cos(phi)*w1*w2))/2
    T1_series.append(T1)
    T2_series.append(T2)

    U1 = -m1*g*l1*np.cos(theta1)
    U2 = -m2*g*(l1*np.cos(theta1)+l2*np.cos(theta2))
    U1_series.append(U1)
    U2_series.append(U2)

    H = T1 + T2 + U1 + U2
    H_series.append(H)


# thetas = [np.pi/6, np.pi/4, np.pi/3, np.pi/2]
# thetas_str = ["pi6", "pi4", "pi3", "pi2"]
# thetas_tex = [R"\pi/6", R"\pi/4", R"\pi/3", R"\pi/2"]
thetas = [np.pi/12, np.pi/24]
thetas_str = ["pi12", "pi24"]
thetas_tex = [R"\pi/12", R"\pi/24"]
dts = np.logspace(-5, -2, num=7, base=10.0)


for i in range(len(thetas)):
    for j in range(len(thetas)):
        theta10 = thetas[i]
        theta20 = thetas[j]
        theta10_str = thetas_str[i]
        theta20_str = thetas_str[j]
        theta10_tex = thetas_tex[i]
        theta20_tex = thetas_tex[j]

        fig, ax = plt.subplots()

        phi0 = (theta10-theta20) % (2*np.pi)

        T10 = m1*(l1**2)*(w10**2)/2
        T20 = (m2*((l1**2)*(w10**2)+(l2**2)*(w20**2)
                   + 2*l1*l2*np.cos(phi0)*w10*w20))/2
        U10 = -m1*g*l1*np.cos(theta10)
        U20 = -m2*g*(l1*np.cos(theta10)+l2*np.cos(theta20))
        exact_H = T10 + T20 + U10 + U20

        y = []

        for dt in dts:
            steps = int((t_end - t_start) / dt)
            theta1_series = deque([theta10])
            theta2_series = deque([theta20])
            w1_series = deque([w10])
            w2_series = deque([w20])

            for step in range(steps):
                theta1 = theta1_series[-1]
                theta2 = theta2_series[-1]
                w1 = w1_series[-1]
                w2 = w2_series[-1]
                RungeKutta42(theta1, theta2, w1, w2, dt)
                # print(theta1, theta2, w1, w2, dt, 'next')

            y.append(np.log(np.abs(1 - H_series[-1]/exact_H)))

        ax.plot(dts, y, label=R"$\theta_1 = {0}, \theta_2 = {1}$"
                .format(theta10_tex, theta20_tex))
        ax.semilogx()
        # ax.semilogy()
        ax.grid()
        ax.legend(fontsize="14")
        ax.set_xlabel(R"$ \Delta t$", fontsize="14")
        ax.set_ylabel(R"$\log|1-E/E_0|$", fontsize="14")
        ax.set_title('Evaluation of RK4 (Double Pendulum)', fontsize="16")
        fig.tight_layout()
        fig.savefig('./figure/RK42/evaluation1/{6}_{7}_{0}-{1}-{2}-{3}{4}{5}.jpeg'
                    .format(dt_now.year, dt_now.month, dt_now.day, dt_now.hour,
                            dt_now.minute, dt_now.second,
                            theta10_str, theta20_str))
