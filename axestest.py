import numpy as np
import matplotlib.pyplot as plt

# 创建一个包含2行2列子图的图形
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# 生成x数据
x = np.linspace(0, 2 * np.pi, 100)

# 在第一个子图中绘制正弦波
axs[0, 0].plot(x, np.sin(x), label='sin(x)')
axs[0, 0].set_title('Sin Wave')
axs[0, 0].legend()

# 在第二个子图中绘制余弦波
axs[0, 1].plot(x, np.cos(x), label='cos(x)')
axs[0, 1].set_title('Cos Wave')
axs[0, 1].legend()

# 在第三个子图中绘制线性函数
axs[1, 0].plot(x, x, label='y = x')
axs[1, 0].set_title('Linear Function')
axs[1, 0].legend()

# 在第四个子图中绘制二次函数
axs[1, 1].plot(x, x**2, label='y = x^2')
axs[1, 1].set_title('Quadratic Function')
axs[1, 1].legend()

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()