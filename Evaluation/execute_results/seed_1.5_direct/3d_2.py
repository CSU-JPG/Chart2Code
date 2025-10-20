import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Data for Targets (orange line with markers)
targets_x = np.array([0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
targets_y = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
targets_z = np.array([1.2, 1.3, 1.1, 1.0, 0.3, 1.4, 0.4, 1.1])

# Data for Predictions (blue line with markers)
predictions_x = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
predictions_y = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
predictions_z = np.array([1.0, 1.1, 1.0, 0.9, 0.8, 1.2, 0.7, 0.6, 0.5])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Targets
ax.plot(targets_x, targets_y, targets_z, color='orange', marker='o', label='Targets')

# Plot Predictions
ax.plot(predictions_x, predictions_y, predictions_z, color='blue', marker='o', label='Predictions')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add legend
ax.legend()

# Enable grid
ax.grid(True)

plt.show()