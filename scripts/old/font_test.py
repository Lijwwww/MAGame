import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = os.getcwd() + '/assets/Fonts/NotoSansCJK-Black.ttc'  # 字体路径
prop = font_manager.FontProperties(fname=font_path)  # 创建字体属性对象

print("Loaded font family:", prop.get_name())  # 检查字体名称是否正确

# 设置 Matplotlib 采用该字体
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题

# 测试绘图
plt.figure()
plt.title("中文测试：数据分析",)
plt.xlabel("时间", fontproperties=prop)
plt.ylabel("数值", fontproperties=prop)
plt.savefig(os.getcwd() + '/screenshots/test.png')

