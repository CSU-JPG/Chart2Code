import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Generate data for the blue curve (x, y)
theta = np.linspace(0, 2 * np.pi, 100)
x_curve = np.sin(theta)
y_curve = np.cos(theta)
z_curve = np.zeros_like(x_curve)  # z is 0 for the curve in (x,y)

# Generate data for red points (x, z)
x_red = np.linspace(0, 1, 20)
z_red = np.linspace(1, 0, 20)
y_red = np.zeros_like(x_red)  # y is 0 for points in (x,z)

# Generate data for green points (assuming they lie on a circle in x-z plane)
theta_green = np.linspace(0, np.pi, 20)
x_green = np.sin(theta_green)
z_green = np.cos(theta_green)
y_green = np.zeros_like(x_green)  # y is 0 for these green points (adjusted visually)

# Generate data for black points (assuming a curve in x-y plane, z=0.5)
x_black = np.linspace(0.1, 0.9, 20)
y_black = np.linspace(0.1, 0.9, 20)
z_black = 0.5 * np.ones_like(x_black)

# Generate data for blue points (assuming a curve in y-z plane, x=0.5)
y_blue = np.linspace(0, 1, 20)
z_blue = np.linspace(0, 1, 20)
x_blue = 0.5 * np.ones_like(y_blue)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot blue curve (x, y)
ax.plot(x_curve, y_curve, z_curve, color='blue', label='curve in (x, y)')

# Plot red points (x, z)
ax.scatter(x_red, y_red, z_red, color='red', label='points in (x, z)')

# Plot green points (visually inferred)
ax.scatter(x_green, y_green, z_green, color='green')

# Plot black points (visually inferred)
ax.scatter(x_black, y_black, z_black, color='black')

# Plot blue points (visually inferred)
ax.scatter(x_blue, y_blue, z_blue, color='blue')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add legend
ax.legend()

# Enable grid lines
ax.xaxis._axinfo["grid"].update({"visible": True, "color": "lightgray"})
ax.yaxis._axinfo["grid"].update({"visible": True, "color": "lightgray"})
ax.zaxis._axinfo["grid"].update({"visible": True, "color": "lightgray"})

plt.show()