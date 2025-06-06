import matplotlib.pyplot as plt
import pandas as pd

# Import data from CSV
file_path = 'bottleneck_distances_2.csv'  # Change this to your CSV file path
data = pd.read_csv(file_path)

# Assuming the CSV has two columns, we extract them
d_W = data.iloc[:, 0]  # First column
d_B = data.iloc[:, 1]  # Second column

# Plot first scatterplot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(d_W, d_B, c='blue', alpha=0.5, label='Data Points')
axs[0].set_xlabel('$d_W$ Image Space')
axs[0].set_ylabel('Bottleneck Distance Persistence Diagram')
axs[0].set_title('Lipschitz Continuity Experiment')

axs[1].scatter(d_W, d_W/d_B, c='red', alpha=0.5, label='Ratio Data')
axs[1].set_xlabel('$d_W$ Image Space')
axs[1].set_ylabel('$d_W$ Image Space / $d_B$ Persistence Diagram')
axs[1].set_title('Ratio Plot')

plt.tight_layout()
plt.show()
