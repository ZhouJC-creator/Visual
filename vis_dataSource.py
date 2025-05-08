# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
#
# # 设置中文宋体和英文Times New Roman
# chinese_font = FontProperties(family='SimSun')  # 中文字体
# english_font = FontProperties(family='Times New Roman')  # 英文字体
#
# # 数据
# labels = ['PVD', 'ATLDSD', 'PPCD2020', 'PPCD2021']
# sizes = [370, 1431, 1730, 11051]
# colors = ['#5B7DB1', '#4876A3', '#7B9BC5', '#A6BFE2']
#
# # 画环形饼图
# fig, ax = plt.subplots(figsize=(12, 8))  # 设置图片尺寸
# wedges, texts, autotexts = ax.pie(
#     sizes, labels=labels, autopct=lambda p: f'{p:.1f}%, {int(p * sum(sizes) / 100)}',
#     colors=colors, startangle=90, wedgeprops={'width': 0.3}
# )
#
# # 设置标签字体
# for text, autotext in zip(texts, autotexts):
#     text.set_fontproperties(english_font)  # 英文标签
#     autotext.set_fontproperties(english_font)  # 百分比文本
#     text.set_fontsize(10)
#     autotext.set_fontsize(9)
#
# # 设置标题
# plt.title('AppleLeaf9', fontproperties=english_font, fontsize=16)
#
# # 图例放在图片下方
# plt.legend(wedges, labels, title="类别", title_fontproperties=chinese_font,
#            loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
#
# # 调整布局
# plt.subplots_adjust(bottom=0.2)
#
# plt.savefig(f"./output/dataset/dataSource.png",dpi=600)  # 保存为PNG格式的图片
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文宋体和英文Times New Roman
chinese_font = FontProperties(family='SimSun')  # 中文字体
english_font = FontProperties(family='Times New Roman')  # 英文字体

# 数据
labels = ['PVD', 'ATLDSD', 'PPCD2020', 'PPCD2021']
sizes = [370, 1431, 1730, 11051]
colors = ['#5B7DB1', '#4876A3', '#7B9BC5', '#A6BFE2']

# 画环形饼图
fig, ax = plt.subplots(figsize=(12, 8))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct=lambda p: f'{p:.1f}%, {int(p * sum(sizes) / 100)}',
    colors=colors, startangle=90, wedgeprops={'width': 0.3}
)

# 在环形饼图中央添加标题
plt.text(0, 0, 'AppleLeaf9', fontproperties=english_font, fontsize=18, ha='center', va='center')

# 设置标签字体
for text, autotext in zip(texts, autotexts):
    text.set_fontproperties(english_font)
    autotext.set_fontproperties(english_font)
    text.set_fontsize(12)
    autotext.set_fontsize(10)

# 调整图例位置（往上移动）
plt.legend(wedges, labels, title="类别", title_fontproperties=chinese_font,
           loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)

# 调整布局，增加下边距，避免图例被裁剪
plt.subplots_adjust(bottom=0.3)

# 保存图片
plt.savefig(f"./output/dataset/dataSource.png",dpi=300,bbox_inches='tight')  # 保存为PNG格式的图片
