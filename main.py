import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters
R1 = 32
R2 = 20
n = 200

# Define functions for cylinders
def Ok(u):
    return np.array([np.zeros_like(u), np.sin(u)*R1, np.zeros_like(u)])[:, np.newaxis]

def Ok1(u1):
    return np.array([np.cos(u1)*R2, np.sin(u1)*R2, np.zeros_like(u1)])[:, np.newaxis]

def Ok2(v2):
    return np.array([np.zeros_like(v2), np.zeros_like(v2), v2])[:, np.newaxis]

def cylinder_ineq(x, y, z, R):
    return x**2 + y**2 - R**2

def T_shape_ineq(x, y, z):
    return np.maximum(-z, cylinder_ineq(x-16, y, z, 20))

# Define functions for surfaces
def Z(u, v):
    w = np.array([1, 0, 0])[:, np.newaxis, np.newaxis]
    return (Ok(u) + w*v)[:, 0, :]

def Z1(u1, v1):
    w1 = np.array([0, 0, 1])[:, np.newaxis, np.newaxis]
    return (Ok1(u1) + w1*v1)[:, 0, :]

def Z2(v2, w2):
    u2 = np.array([0, 1, 0])[:, np.newaxis, np.newaxis]
    return (Ok2(v2) + u2*w2)[:, 0, :]

# Define function for intersection line
def line_of_intersection(u, R1, R2):
    return np.array([np.cos(u)*R2, np.sin(u)*R2, R1*np.sqrt(1-(R2/R1)**2)*np.sin(u)**2])

# Define function for integral
def integrand(u, R1, R2):
    return np.sqrt(1 + ((R2**2) * np.sin(2*u)**2)/(4 * ((R1**2) - ((R2**2) * np.sin(u)**2))))

# Define function for length of line of intersection
def line_length(R1, R2):
    integral, error = quad(integrand, 0, 2*np.pi, args=(R1, R2))
    return R2 * integral

# Compute surfaces
u_mesh, v_mesh = np.meshgrid(np.linspace(0, 2*np.pi, 41), np.linspace(-2, 2, 101))
u1_mesh, v1_mesh = np.meshgrid(np.linspace(0, 2*np.pi, 41), np.linspace(2.6, 4, 101))
v2_mesh, w2_mesh = np.meshgrid(np.linspace(-20, 20, 101), np.linspace(-20, 20, 101))

Z_0 = Z(u_mesh, v_mesh)
Z1_0 = Z1(u1_mesh, v1_mesh)
Z2_0 = Z2(v2_mesh, w2_mesh)

X = Z_0[0]
Y = Z_0[1]
Z = Z_0[2]

X1 = Z1_0[0]
Y1 = Z1_0[1]
Z1 = Z1_0[2]

# Compute line of intersection
u = np.linspace(0, 2*np.pi, n)
line = line_of_intersection(u, R1, R2)

# Plot surfaces
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, alpha=0.5, color='blue')
ax.plot_surface(X1, Y1, Z1, alpha=0.5, color='red')
ax.plot_surface(v2_mesh, Z2_0[1], Z2_0[2], alpha=0.5, color='green')

# Plot line of intersection
line_pts = line_of_intersection(u, R1, R2)
ax.plot(line_pts[0], line_pts[1], line_pts[2], 'k-', linewidth=2)

# Set plot parameters
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_zlim(-40, 40)

plt.show()