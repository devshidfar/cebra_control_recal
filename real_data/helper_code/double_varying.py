import numpy as np
import matplotlib.pyplot as plt

# Torus parameters
R = 3.5      # radius of the great circle
r = 0.7     # radius of the tube (smaller = flatter, wider-looking donut)
n_theta = 100
n_phi = 60

theta = np.linspace(0, 2*np.pi, n_theta)  # angle around great circle
phi = np.linspace(0, 2*np.pi, n_phi)      # angle around small circle
Theta, Phi = np.meshgrid(theta, phi)

X = (R + r*np.cos(Phi)) * np.cos(Theta)
Y = (R + r*np.cos(Phi)) * np.sin(Theta)
Z = r * np.sin(Phi)

fig = plt.figure(figsize=(16, 7))

# --- Spatial representation: varies around the GREAT circle (theta) ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
sc1 = ax1.scatter(X.ravel(), Y.ravel(), Z.ravel(),
                   c=Theta.ravel(), cmap='jet', s=8)
cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.5, pad=0.2)
ax1.set_title('Varying Spatial Neural Representation', fontweight='bold')
ax1.set_xlabel('X-axis'); ax1.set_ylabel('Y-axis'); ax1.set_zlabel('Z-axis')
ax1.view_init(elev=30, azim=-37.5)
ax1.set_box_aspect([1, 1, 2*r/R])

# --- Audio representation: varies around the SMALL circle (phi) ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
sc2 = ax2.scatter(X.ravel(), Y.ravel(), Z.ravel(),
                   c=np.sin(Phi.ravel()), cmap='viridis', s=8)
cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.5, pad=0.2)
ax2.set_title('Varying Audio Neural Representation', fontweight='bold')
ax2.set_xlabel('X-axis'); ax2.set_ylabel('Y-axis'); ax2.set_zlabel('Z-axis')
ax2.view_init(elev=30, azim=-37.5)
ax2.set_box_aspect([1, 1, 2*r/R])

plt.subplots_adjust(wspace=0.4)
plt.savefig('torus_representations.svg', format='svg', bbox_inches='tight')
plt.show()