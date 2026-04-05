import warnings
import logging
# ====== 必须在导入 matplotlib 之前设置 ======
# 屏蔽 warnings 模块的字体警告
warnings.filterwarnings('ignore', message='.*findfont.*')
warnings.filterwarnings('ignore', message='.*font.*')
# 屏蔽 logging 系统的字体警告（关键！）
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
# ====== 之后再导入 matplotlib ======

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys

# 应用 Nature 样式
plt.style.use('../nature.mplstyle')  # 确保 nature.mplstyle 文件在同一目录下

# 读取force数据
force_file = './force_train.out'
force_data = np.loadtxt(force_file)
# Force数据：DFT和NEP的三个方向
data_fx = force_data[:, 3]; data_fy = force_data[:, 4]; data_fz = force_data[:, 5]  # DFT
pred_fx = force_data[:, 0]; pred_fy = force_data[:, 1]; pred_fz = force_data[:, 2]  # NEP

# 读取energy数据
energy_file = './energy_train.out'
energy_data = np.loadtxt(energy_file, comments='#')
data_energy = energy_data[:, 1]  # DFT energy
pred_energy = energy_data[:, 0]  # NEP energy

# 计算RMSE的函数
def calculate_rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values)**2))

# 计算R²的函数
def calculate_r2(true_values, predicted_values):
    ss_res = np.sum((true_values - predicted_values) ** 2)  # 残差平方和
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)  # 总平方和
    r2 = 1 - (ss_res / ss_tot)  # R²值
    return r2

# 创建包含两个子图的图像
fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.05), dpi=600)  # 2倍单栏宽度
fig.subplots_adjust(wspace=0.3)  # 调整子图间距

# 绘制Force y方向子图
ax_force = axes[0]

# 计算三个方向Force的RMSE和R²，然后求平均
rmse_fx = calculate_rmse(data_fx, pred_fx)
rmse_fy = calculate_rmse(data_fy, pred_fy)
rmse_fz = calculate_rmse(data_fz, pred_fz)
rmse_force_avg = (rmse_fx + rmse_fy + rmse_fz) / 3

r2_fx = calculate_r2(data_fx, pred_fx)
r2_fy = calculate_r2(data_fy, pred_fy)
r2_fz = calculate_r2(data_fz, pred_fz)
r2_force_avg = (r2_fx + r2_fy + r2_fz) / 3

# 绘制Force y方向散点图
ax_force.scatter(data_fy, pred_fy,
          marker='o',
          s=4,  # 小标记符合 Nature 样式
          linewidths=0.6)

# 绘制参考线
ax_force.plot([-10, 10], [-10, 10], '-', color='gray', linewidth=1.0)

# 设置Force坐标轴
ax_force.set_xlim([-10, 10])
ax_force.set_xticks([-10, -5, 0, 5, 10])
ax_force.set_ylim([-10, 10])
ax_force.set_yticks([-10, -5, 0, 5, 10])

# 设置Force标签
ax_force.set_xlabel(r"DFT (eV/Å)")
ax_force.set_ylabel(r"NEP (eV/Å)")

# 添加Force图例
ax_force.legend(['Force'], frameon=False, loc='upper left', fontsize=8.0)

# 在右下角添加Force平均RMSE和R²文本
# ✅ 修改：乘以1000转换为meV，单位改为meV/Å
ax_force.text(0.98, 0.08, f'RMSE={rmse_force_avg * 1000:.2f} meV/$\\mathrm{{\\AA}}$',
        transform=ax_force.transAxes, fontsize=7.0,
        verticalalignment='bottom', horizontalalignment='right')

# 将 R² 转换为百分比，保留 1 位小数
ax_force.text(0.98, 0.02, f'$R^2$={r2_force_avg * 100:.1f}%',
        transform=ax_force.transAxes, fontsize=7.0,
        verticalalignment='bottom', horizontalalignment='right')

# 设置Force刻度参数
ax_force.tick_params(which='both', direction='in', right=True, top=True)

# 绘制Energy子图
ax_energy = axes[1]

# 计算Energy RMSE和R²
rmse_energy = calculate_rmse(data_energy, pred_energy)
r2_energy = calculate_r2(data_energy, pred_energy)

# 绘制Energy散点图
ax_energy.scatter(data_energy, pred_energy,
          marker='o',
          s=4,  # 小标记符合 Nature 样式
          c='C2',
          linewidths=0.6)

# 绘制参考线 y=x
ax_energy.plot([-10, -1], [-10, -1], '-', color='gray', linewidth=1.0)

# 设置Energy坐标轴范围和刻度
ax_energy.set_xlim([-4.8, -4.2])
ax_energy.set_xticks([-4.8,-4.6, -4.4,-4.2])
ax_energy.set_ylim([-4.8, -4.2])
ax_energy.set_yticks([-4.8, -4.6, -4.4,-4.2])

# 设置Energy标签
ax_energy.set_xlabel(r"DFT (eV/atom)")
ax_energy.set_ylabel(r"NEP (eV/atom)")

# 添加Energy图例
ax_energy.legend(['Energy'], frameon=False, loc='upper left', fontsize=8.0)

# 在右下角添加Energy RMSE和R²文本
# ✅ 修改：乘以1000转换为meV，单位改为meV/atom
ax_energy.text(0.98, 0.08, f'RMSE={rmse_energy * 1000:.2f} meV/atom',
        transform=ax_energy.transAxes, fontsize=7.0,
        verticalalignment='bottom', horizontalalignment='right')

# 将 R² 转换为百分比，保留 1 位小数
ax_energy.text(0.98, 0.02, f'$R^2$={r2_energy * 100:.1f}%',
        transform=ax_energy.transAxes, fontsize=7.0,
        verticalalignment='bottom', horizontalalignment='right')

# 设置Energy刻度参数
ax_energy.tick_params(which='both', direction='in', right=True, top=True)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig("force_energy_combined.png", dpi=600, bbox_inches='tight')
