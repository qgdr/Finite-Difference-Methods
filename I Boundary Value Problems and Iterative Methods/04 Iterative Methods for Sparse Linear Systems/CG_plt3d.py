import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    [1.5, 0.4, 0.2],
    [0.4, 1.5, 0.8],
    [0.2, 0.8, 1.5]
])
eigval, eigvec = np.linalg.eig(A)
B = np.diag(np.sqrt(eigval)) @ eigvec.T

f = np.array([0, 0, 0])
u0 = np.array([1, -.8, 1])

r0 = f-A@u0
p = [r0]

u = [u0]
r = [r0]
w = []
a = []
b = []

for k in range(3):
    w.append(A @ p[k])
    a.append( (r[k] @ r[k]) / (p[k] @ w[k]) )

    u.append(u[k]+a[k]*p[k])
    r.append(r[k]-a[k]*w[k])

    b.append((r[k+1] @ r[k+1]) / (r[k] @ r[k]))
    p.append(r[k+1]+b[k]*p[k])

###################
ph = np.linspace(0, 2*np.pi, 200)
th = np.linspace(0, np.pi, 100)

ph, th = np.meshgrid(ph, th)
sphere = np.array([
    np.cos(ph)*np.sin(th),
    np.sin(ph)*np.sin(th),
    np.cos(th),
])
sh = sphere.shape


unit_contour = np.linalg.solve(B, sphere.reshape([3, -1])).reshape(sh)

fig = plt.figure()
ax = plt.axes(projection='3d')

mult = [np.sqrt(ui @ A @ ui) for ui in u]

for m in mult:
    ax.plot_surface(m*unit_contour[0], m*unit_contour[1], m*unit_contour[2], alpha=0.3, cmap='Blues_r')    # 等高线

# ax.scatter3D(eigvec[0], eigvec[1], eigvec[2], c='red')
print(u)
u = np.array(u).T
ax.plot3D(u[0], u[1], u[2], '-or')       # 点

plt.axis('equal')
plt.show()
