import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "subject1_normal_trial1_pressure.csv"  # change if needed

data = pd.read_csv("Pressure_Data\\subject1_steppage_trial16_pressure.csv", header=None)
plt.figure(figsize=(6, 12))  # width, height

print("Shape:", data[::-1].shape)

plt.imshow(data[::-1])
plt.title("Raw Pressure")
plt.colorbar()
plt.show()