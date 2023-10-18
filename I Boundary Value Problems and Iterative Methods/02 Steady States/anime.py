import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def G(m, theta, alpha, beta):
    h = 2*np.pi/(m+1)
    A = np.diag(-2*np.ones(m)) + np.diag(np.ones(m-1), -1) + np.diag(np.ones(m-1), 1)
    A = A/h**2
    res =  A @ theta + np.sin(theta)
    res = np.reshape(res, (-1,))
    res[0] += alpha/h**2
    res[-1] += beta/h**2
    return res

def J(m, theta):
    h = 2*np.pi/(m+1)
    jacobi = np.diag(np.cos(theta))
    A = np.diag(-2*np.ones(m)) + np.diag(np.ones(m-1), -1) + np.diag(np.ones(m-1), 1)
    return jacobi + A/h**2


alpha=beta=0.7
m = 80 -1
ti = np.linspace(0, 2*np.pi, m+2)


theta = 0.7*np.cos(ti) + 0.5*np.sin(ti)


for i in range(8):
    delta = np.linalg.solve(J(m, theta[1:-1]), -G(m, theta[1:-1], alpha, beta))
    theta[1:-1] += delta


theta1 = 0.7+np.sin(ti/2)

for i in range(8):
    delta = np.linalg.solve(J(m, theta1[1:-1]), -G(m, theta1[1:-1], alpha, beta))
    theta1[1:-1] += delta



fig = plt.figure()
ax = fig.subplots()

t = np.linspace(0, 2*np.pi)
ax.plot(np.cos(t), np.sin(t), '--')
point1, = ax.plot([0, np.sin(theta[0])], [0, -np.cos(theta[0])], '--ob')
point2, = ax.plot([0, np.sin(theta1[0])], [0, -np.cos(theta1[0])], '--or')

def update(i):
    point1.set_data([0, np.sin(theta[i])], [0, -np.cos(theta[i])])
    point2.set_data([0, np.sin(theta1[i])], [0, -np.cos(theta1[i])])
    return

ani=FuncAnimation(fig, func=update, interval=20)
plt.axis('equal')
plt.show()
