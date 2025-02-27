import matplotlib.pyplot as plt
import numpy as np

# Data
difficulty_levels = ["Easy", "Medium", "Hard", "Expert"]
steps = [25, 44, 6320, 426]
time = [0.0183, 0.0283, 4.2489, 0.2806]
memory = [17.53, 20.07, 16.8, 16.1]
backtracks = [0, 4, 6265, 362]

x = np.arange(len(difficulty_levels))

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot steps and backtracks on left y-axis
ax1.set_xlabel("Difficulty")
ax1.set_ylabel("Steps / Backtracks", color="tab:blue")
ax1.bar(x - 0.2, steps, width=0.4, label="Steps", color="tab:blue", alpha=0.6)
ax1.bar(x + 0.2, backtracks, width=0.4, label="Backtracks", color="tab:purple", alpha=0.6)
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Create second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel("Time (s) / Memory (KB)", color="tab:red")
ax2.plot(x, time, marker="o", linestyle="-", color="tab:red", label="Time (s)")
ax2.plot(x, memory, marker="s", linestyle="--", color="tab:orange", label="Memory (KB)")
ax2.tick_params(axis="y", labelcolor="tab:red")

# Add labels and legend
ax1.set_xticks(x)
ax1.set_xticklabels(difficulty_levels)
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
plt.title("DFS Sudoku Solver Performance by Difficulty")

# Show plot
plt.show()
