import matplotlib.pyplot as plt
from matplotlib.patches import Arrow

# 创建一个图形对象和子图
fig, ax = plt.subplots()

# 设置字体大小和颜色
font_size = 25
font_color = 'yellow'

# 设置背景颜色
background_color = 'green'

# 绘制箭头路标，设置填充颜色和边框颜色
arrow = Arrow(0.2, 0.2, 0.6, 0.6, width=0.5, edgecolor='yellow', facecolor='red', linewidth=3)
ax.add_patch(arrow)

# 添加注释文本，设置字体大小和颜色
ax.text(0.5, 0.9, 'RIGHT', ha='center', va='center', fontsize=font_size, color=font_color)

# 设置图像范围
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# 设置整个图形对象的背景颜色
fig.patch.set_facecolor(background_color)

# 隐藏坐标轴
ax.axis('off')

# 显示图像
plt.show()
