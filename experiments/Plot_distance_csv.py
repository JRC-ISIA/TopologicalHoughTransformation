import matplotlib.pyplot as plt
import pandas as pd

# Import data from CSV
file_path = 'bottleneck_distances_2.csv'  # Change this to your CSV file path
data = pd.read_csv(file_path)

# Assuming the CSV has two columns, we extract them
x = data.iloc[:, 0]  # First column
y = data.iloc[:, 1]  # Second column


# Plot first scatterplot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x, y, c='blue', alpha=0.5, label='Data Points')
plt.xlabel('1-Wasserstein Distance Image Space')
plt.ylabel('Bottleneck Distance Persistence Diagram')
plt.title('Lipschitz Continuity Experiment')

# Plot second scatterplot
plt.subplot(1, 2, 2)
plt.scatter(x, y/x, c='red', alpha=0.5, label='Ratio Data')
plt.xlabel('1-Wasserstein Distance Image Space')
plt.ylabel('1-Wasserstein Distance Image Space / Bottleneck Distance Persistence Diagram')
plt.title('Ratio Plot')

plt.tight_layout()
plt.show()

print(y/x)
