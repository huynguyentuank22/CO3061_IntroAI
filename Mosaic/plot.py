import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu từ kết quả thực nghiệm
algorithms = ["DFS (Run 1)", "DFS (Run 2)", "BFS (Run 1)", "BFS (Run 2)"]
steps = [119892, 45144, 250962, 176214]
time_elapsed = [9.088218688964844, 3.273099422454834, 9.696228742599487, 9.538520812988281]
memory_usage = [16.34375, 16.33203125, 22.828125, 22.8359375]

x = np.arange(len(algorithms))  # Vị trí trên trục X
width = 0.3  # Độ rộng của các cột

# Vẽ biểu đồ
fig, ax = plt.subplots(3, 1, figsize=(8, 12))

# Biểu đồ số bước thực hiện
ax[0].bar(x, steps, width, color=['blue', 'blue', 'red', 'red'])
ax[0].set_ylabel("Steps")
ax[0].set_title("So sánh số bước thực hiện")
ax[0].set_xticks(x)
ax[0].set_xticklabels(algorithms, rotation=20)

# Biểu đồ thời gian chạy
ax[1].bar(x, time_elapsed, width, color=['blue', 'blue', 'red', 'red'])
ax[1].set_ylabel("Time (seconds)")
ax[1].set_title("So sánh thời gian thực thi")
ax[1].set_xticks(x)
ax[1].set_xticklabels(algorithms, rotation=20)

# Biểu đồ sử dụng bộ nhớ
ax[2].bar(x, memory_usage, width, color=['blue', 'blue', 'red', 'red'])
ax[2].set_ylabel("Memory (MB)")
ax[2].set_title("So sánh mức tiêu thụ bộ nhớ")
ax[2].set_xticks(x)
ax[2].set_xticklabels(algorithms, rotation=20)

plt.tight_layout()
plt.show()
