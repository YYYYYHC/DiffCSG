
import matplotlib.pyplot as plt
# Data
n_triangles = [32, 128, 512, 2048, 8192]
time_taken = [0.004, 0.006, 0.008, 0.03, 0.5]
time_taken_ms = [t * 1000 for t in time_taken]
# Calculate the square of the number of triangles for the x-axis labels
triangle_pairs_labels = [f"{n//2}^2" for n in n_triangles]

# Plotting the original data with n^2 values as the labels for the x-axis
plt.figure(figsize=(10, 6))
plt.plot(n_triangles, time_taken_ms, marker='o', linestyle='-', color='black',markerfacecolor='white', markeredgewidth=1.5, markersize=10)

# Set x-axis to logarithmic and custom ticks only at data points with squared labels
plt.xscale('log')
plt.xticks(n_triangles, labels=triangle_pairs_labels)

plt.xlabel('Number of Triangle Pairs (n^2)')
plt.ylabel('Time (ms)')
plt.title('Time taken vs Number of Triangle Pairs')

# Ensure no grid is present and only ticks at data points
plt.grid(False)
plt.minorticks_off()

# Show plot
plt.savefig('drawPlot.png')
