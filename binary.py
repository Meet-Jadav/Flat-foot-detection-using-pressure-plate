import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "subject1_normal_trial1_pressure.csv"  # change if needed

data = pd.read_csv("Pressure_Data\\subject1_steppage_trial16_pressure.csv", header=None)
plt.figure(figsize=(6, 12))  # width, height

data = data[::-1]  # flip vertically for better visualization

print("Shape:", data[::-1].shape)

threshold = 5  # you can tune later

binary = (data > threshold).astype(int)

#to show binary image
#plt.imshow(binary[::-1], cmap="gray")
#plt.title("Binary (Contact Map)")
#plt.show()

#extract only one foot 

from scipy.ndimage import label

labeled, num_features = label(binary)

print("Number of regions:", num_features)

regions = []

for i in range(1, num_features+1):
    size = np.sum(labeled == i)
    regions.append((i, size))

regions.sort(key=lambda x: x[1], reverse=True)

largest_label = regions[0][0]

foot = (labeled == largest_label).astype(int)

plt.imshow(foot, cmap='gray')
plt.title("Single Foot")
plt.show()  