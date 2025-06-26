from mplsoccer import Pitch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2


pitch = Pitch(pitch_type="uefa" , pitch_color='grass', line_color='white', stripe=True)

fig, ax = pitch.draw()

#  绘制传球路线（示例数据）
x =  [10, 25]
y = [50, 30]
end_x = [25, 40]
end_y = [30, 20]
pitch.arrows(x, y, end_x, end_y, ax=ax)


# 添加球员位置
player_data = {
    'player_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'player_name': ['守门员', '右后卫', '中后卫1', '中后卫2', '左后卫', 
                  '右中场', '中前卫', '左中场', '前腰', '前锋1', '前锋2'],
    'x': [10, 25, 30, 30, 25, 40, 50, 40, 60, 70, 70],
    'y': [50, 30, 50, 70, 70, 20, 50, 60, 50, 40, 60],
    'team': ['A'] * 11,   # 所有球员属于同一队
    'pos':['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'LM', 'CAM', 'ST', 'ST']
}


#标注属性
for idx, pid in enumerate(player_data['player_id']):
    scatter = pitch.scatter(
    player_data['x'][idx], 
    player_data['y'][idx], 
    s=300,  # 点的大小
    c='red',  # 颜色
    edgecolors='black',  # 边框颜色
    linewidth=2,  # 边框宽度
    alpha=0.9,  # 透明度
    ax=ax)
    pitch.annotate(str(pid), (player_data['x'][idx], player_data['y'][idx]), va='center', ha='center', color='white', fontweight='bold', ax=ax)
    pitch.annotate(player_data['pos'][idx], (player_data['x'][idx], player_data['y'][idx]-5), va='center', ha='center', color='black', fontsize=9, ax=ax)



# plt.savefig('1.jpg')

# 渲染并获取图像数据
fig.canvas.draw()

# 获取RGBA数据并转换为RGB
data = np.asarray(fig.canvas.buffer_rgba())
data = cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2BGR)  # 丢弃alpha通道
plt.close(fig)

cv2.imwrite('1.jpg', data)