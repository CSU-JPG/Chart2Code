import matplotlib.pyplot as plt
import numpy as np

# Data for Posture1 (olive, line with o markers)
posture1_x = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
posture1_y = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
posture1_z = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.2, 0.4, 0.6]

# Data for Posture2 (green, line with o markers)
posture2_x = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
posture2_y = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
posture2_z = [0.6, 0.8, 1.0, 1.2, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Posture1
ax.plot(posture1_x, posture1_y, posture1_z, color='olive', marker='o', label='Posture1')

# Plot Posture2
ax.plot(posture2_x, posture2_y, posture2_z, color='green', marker='o', label='Posture2')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add legend
ax.legend()

# Set grid and background
ax.xaxis._axinfo["grid"].update({"color": (1, 1, 1, 0.5), "linestyle": "--"})
ax.yaxis._axinfo["grid"].update({"color": (1, 1, 1, 0.5), "linestyle": "--"})
ax.zaxis._axinfo["grid"].update({"color": (1, 1, 1, 0.5), "linestyle": "--"})

plt.show()