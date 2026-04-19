import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
data = pd.read_csv("Pressure_Data\\subject1_normal_trial1_pressure.csv", header=None)

# Plot heatmap
plt.figure(figsize=(6, 12))  # width, height
sns.heatmap(data[::-1], cmap="jet", square=False, vmin=0, vmax=100)
plt.gca().set_aspect('auto')


plt.show()

# Show plot
plt.title("Heatmap of CSV Data")
plt.show()