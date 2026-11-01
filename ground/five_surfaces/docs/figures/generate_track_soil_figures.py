"""Reproduce documentation figures from the current project geometry/config.

Run from any directory with Python, NumPy and Matplotlib. No Isaac Sim startup
or simulation-state mutation is performed. PNGs are for Markdown, SVGs for print.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / '.runtime/mpl_docs'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch, FancyBboxPatch
import numpy as np

sys.path.insert(0, str(PROJECT / 'scripts'))
from track_motion import CirculatingTracks

ROBOT = json.loads((PROJECT / 'configs/robot_v1.json').read_text())
TRACK = json.loads((PROJECT / 'configs/tracks.json').read_text())
SOIL = json.loads((PROJECT / 'configs/scm.json').read_text())
MOTION = CirculatingTracks(ROBOT, TRACK)
BLUE, ORANGE, GREEN, RED, GRAY = '#2166ac', '#cf741c', '#21825b', '#bb4747', '#5d6b79'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#8994a1', 'text.color': '#182a3a',
    'axes.labelcolor': '#182a3a', 'xtick.color': '#526170', 'ytick.color': '#526170',
    'mathtext.fontset': 'dejavusans', 'svg.hashsalt': 'track-soil-docs-v1',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def save(fig, name):
    fig.savefig(OUTPUT / (name + '.png'), dpi=170, bbox_inches='tight', pad_inches=.15)
    fig.savefig(OUTPUT / (name + '.svg'), bbox_inches='tight', pad_inches=.15,
                metadata={'Date': None, 'Creator': 'Project track/soil documentation generator'})
    plt.close(fig)


def arrow(ax, a, b, color=BLUE, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=14,
                               lw=1.7, color=color, **kw))


def dimension(ax, a, b, label, offset=(0, 0), color=GRAY, size=10):
    ax.annotate('', xy=a, xytext=b,
                arrowprops={'arrowstyle': '<->', 'color': color, 'lw': 1})
    center = (np.array(a) + b) / 2 + np.array(offset)
    ax.text(*center, label, ha='center', va='center', color=color, fontsize=size,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})


def shoe_polygon(center, angle, length, thickness, z_offset=0):
    p = np.array([[-length/2, -thickness/2], [length/2, -thickness/2],
                  [length/2, thickness/2], [-length/2, thickness/2]])
    p[:, 1] += z_offset
    co, si = np.cos(angle), np.sin(angle)
    return p @ np.array([[co, si], [-si, co]]) + center


def geometry():
    fig = plt.figure(figsize=(12.5, 8.4))
    ax = fig.add_axes([.08, .57, .86, .32])
    centers, angles, _ = MOTION.poses(0)
    s = np.linspace(0, MOTION.perimeter, 800)
    x, z, _, _ = MOTION.path(s)
    ax.plot(x, z, '--', color=BLUE, lw=1, label='履带板中心路径')
    for c, angle in zip(centers, angles):
        ax.add_patch(Polygon(shoe_polygon(c[[0, 2]], angle, TRACK['shoe_length_m'],
                                        TRACK['shoe_thickness_m']), fc='#c8d2dd', ec=GRAY, lw=.6))
        ax.add_patch(Polygon(shoe_polygon(c[[0, 2]], angle, TRACK['grouser_length_m'],
                                        TRACK['grouser_height_m'], -.011), fc=ORANGE, ec=ORANGE, lw=.5))
    inner_bottom = ROBOT['tracks']['center_z_m'] - MOTION.radius + TRACK['shoe_thickness_m']/2
    for wx in TRACK['road_wheel_x_m']:
        ax.add_patch(Circle((wx, inner_bottom + TRACK['road_wheel_radius_m']),
                            TRACK['road_wheel_radius_m'], fc='#e5e9ec', ec=GRAY, lw=1.3))
    for wx in (-MOTION.half_straight, MOTION.half_straight):
        ax.add_patch(Circle((wx, -.1), .097, fc='#d9e6ef', ec=BLUE, lw=1.5))
        ax.plot(wx, -.1, 'o', color=BLUE, ms=3)
    for wx in TRACK['return_roller_x_m']:
        ax.add_patch(Circle((wx, -.027), .025, fc='#e5e9ec', ec=GRAY))
    arrow(ax, (-.13, .059), (.13, .059))
    arrow(ax, (.13, -.26), (-.13, -.26))
    ax.text(.15, .059, r'上段 $+u$', va='center', color=BLUE)
    ax.text(.15, -.26, r'下段 $-u$', va='center', color=BLUE)
    ax.annotate('后驱动轮\n外观半径 0.097 m', xy=(-.41, -.10), xytext=(-.60, .12),
                ha='center', fontsize=10, arrowprops={'arrowstyle': '-', 'color': BLUE})
    ax.annotate('前诱导轮', xy=(.41, -.1), xytext=(.52, .115),
                ha='center', fontsize=10, arrowprops={'arrowstyle': '-', 'color': BLUE})
    ax.text(0, -.124, '5 个承重轮；轮心固定', ha='center', va='center', fontsize=9)
    ax.annotate('2 个托带轮', xy=(.18, -.027), xytext=(.23, .115),
                ha='center', fontsize=10, arrowprops={'arrowstyle': '-', 'color': GRAY})
    dimension(ax, (-.41, -.315), (.41, -.315), '两圆心间距 0.820 m')
    ax.axhline(-.22, color=ORANGE, ls=':', lw=.9)
    ax.text(-.60, -.235, '直线段履齿尖端：z = −0.220 m', color=ORANGE, fontsize=9)
    ax.set(xlim=(-.68, .68), ylim=(-.345, .15), xlabel=r'$x_B$ / m', ylabel=r'$z_B$ / m',
           title='(a) 侧视：几何按当前配置绘制，机体外壳省略；正向指令 u > 0')
    ax.set_aspect('equal')
    ax = fig.add_axes([.08, .10, .86, .33])
    ax.add_patch(Rectangle((-.5, -.3), 1., .6, fc='#eaf0f5', ec=GRAY, lw=1.1))
    for y, side in ((.35, '左'), (-.35, '右')):
        ax.add_patch(Rectangle((-.53, y-.08), 1.06, .16, fc='#d0d9e2', ec=BLUE, lw=1.2))
        ax.plot([-.53, .53], [y, y], '--', color=BLUE, lw=.8)
        ax.text(-.64, y, side, va='center', ha='center', color=BLUE)
    ax.plot(0, 0, 'o', color=RED, ms=5)
    ax.text(.03, .04, '质心 / 车体系原点', fontsize=10)
    arrow(ax, (0, 0), (.23, 0))
    arrow(ax, (0, 0), (0, .21))
    ax.text(.25, 0, r'$+x_B$', va='center')
    ax.text(.015, .21, r'$+y_B$')
    dimension(ax, (.64, -.35), (.64, .35), 'd = 0.700 m', offset=(.11, 0))
    dimension(ax, (.40, .27), (.40, .43), 'b = 0.160 m', offset=(-.20, 0))
    ax.text(-.48, -.22, '机体外壳：1.00 × 0.60 m', fontsize=10)
    ax.set(xlim=(-.72, .86), ylim=(-.52, .52), xlabel=r'$x_B$ / m', ylabel=r'$y_B$ / m',
           title='(b) 俯视：左右中心 y = ±0.350 m；矩形为名义水平投影，不表示实际受压面积')
    ax.set_aspect('equal')
    save(fig, '01_track_geometry')


def component():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={'width_ratios': [1, 1.65]})
    ax = axes[0]
    ax.add_patch(Rectangle((-22.5, -7), 45, 14, fc='#d0d9e2', ec=GRAY, lw=1.5))
    ax.add_patch(Rectangle((-6, -15), 12, 8, fc=ORANGE, ec=ORANGE))
    ax.add_patch(Rectangle((-9, 7), 18, 16, fc='#b9d8ef', ec=BLUE))
    ax.text(0, 0, '履带板', ha='center', va='center')
    ax.text(0, 16, '导向齿', ha='center', va='center', fontsize=9)
    ax.annotate('外侧履齿：参与接触', xy=(0, -11), xytext=(-36, -27),
                fontsize=10, arrowprops={'arrowstyle': '-', 'color': ORANGE})
    dimension(ax, (-22.5, 31), (22.5, 31), '45 mm')
    dimension(ax, (-31, -7), (-31, 7), '14', offset=(-5, 0))
    dimension(ax, (31, -15), (31, -7), '8', offset=(5, 0))
    ax.set(xlim=(-43, 43), ylim=(-34, 39), xlabel=r'$x_s$ / mm', ylabel=r'$z_s$ / mm',
           title='(a) 单板侧视：下直线段姿态')
    ax.set_aspect('equal')
    ax = axes[1]
    ax.add_patch(Rectangle((-80, -7), 160, 14, fc='#d0d9e2', ec=GRAY, lw=1.5))
    ax.add_patch(Rectangle((-80, -15), 160, 8, fc=ORANGE, ec=ORANGE))
    ax.add_patch(Rectangle((-8, 7), 16, 16, fc='#b9d8ef', ec=BLUE))
    for sign in (-1, 1):
        ax.add_patch(Rectangle((sign*83-6, -4), 12, 8, fc='#b9dcca', ec=GREEN))
    dimension(ax, (-80, 33), (80, 33), '板宽 b = 160 mm')
    ax.annotate('销轴外观：直径 8 mm，轴向长度 12 mm\n投影位置 x_s = +22.5 mm；不构成物理铰链',
                xy=(83, 0), xytext=(0, -30), fontsize=9, ha='center',
                arrowprops={'arrowstyle': '-', 'color': GREEN})
    ax.text(0, 16, '导向齿：宽 16、高 16 mm', ha='center', fontsize=9)
    ax.set(xlim=(-104, 104), ylim=(-44, 44), xlabel=r'$y_s$ / mm', ylabel=r'$z_s$ / mm',
           title='(b) 沿 x_s 方向的投影')
    ax.set_aspect('equal')
    fig.text(.5, .02, '局部 +z_s 指向履带闭环内侧；灰色板与橙色履齿参与 SCM，蓝色导向齿与绿色销轴仅显示。', ha='center', fontsize=10)
    fig.subplots_adjust(bottom=.20, top=.88, wspace=.24)
    save(fig, '02_shoe_components')


def pipeline():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    boxes = {
        'cmd': (.03, .78, .18, .14, '左右速度指令\n限幅 ±1 m/s'),
        'move': (.28, .78, .20, .14, '0.15 s 一阶带速\n积分行程 / 更新相位'),
        'ray': (.54, .78, .19, .14, '当前板 / 履齿位姿\n4 子采样求交与平均'),
        'normal': (.79, .78, .18, .14, '总压沉 δ\n承载 / 塑性 / 法向阻尼'),
        'vel': (.54, .44, .19, .16, '车身 + 履带板\n接触材料点速度\n含两级转动项'),
        'shear': (.79, .44, .18, .16, '积分剪切向量 j\nJanosi + 阻尼 + 限幅'),
        'sum': (.79, .10, .18, .16, '单元应力 × 覆盖面积\n累加 F 与 r × F'),
        'physx': (.54, .10, .19, .16, 'PhysX 积分\n整车 6 自由度'),
        'state': (.28, .36, .20, .16, '读取车身状态\n位置 / 姿态 / 速度'),
    }
    for key, (x, y, w, h, label) in boxes.items():
        fill = '#fff0dd' if key in ('normal', 'shear', 'sum') else '#eaf2f9'
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008', fc=fill, ec=GRAY, lw=1.2))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=11)
    for a, b in [((.21, .85), (.28, .85)), ((.48, .85), (.54, .85)),
                 ((.73, .85), (.79, .85)), ((.635, .78), (.635, .60)),
                 ((.73, .52), (.79, .52)), ((.88, .78), (.88, .60)),
                 ((.88, .44), (.88, .26)), ((.79, .18), (.73, .18)),
                 ((.54, .18), (.38, .36)), ((.48, .44), (.54, .48)),
                 ((.48, .50), (.54, .79))]:
        arrow(ax, a, b)
    ax.text(.92, .68, r'$p\rightarrow c+p\tan\phi$', ha='center', fontsize=10)
    arrow(ax, (.73, .60), (.80, .78), color=ORANGE, linestyle='--')
    ax.text(.755, .69, r'$v_n$', color=ORANGE, fontsize=10)
    ax.plot([.97, .99, .99], [.85, .85, .18], color=GREEN, lw=1.3)
    arrow(ax, (.99, .18), (.97, .18), color=GREEN)
    ax.text(1.005, .37, '法向压力 p', color=GREEN, fontsize=9, rotation=90)
    ax.text(.42, .68, '车身位姿', ha='center', fontsize=10, color=GRAY)
    arrow(ax, (.64, .015), (.64, .10), color=RED)
    ax.text(.49, .025, '重力由 PhysX 提供', ha='right', color=RED, fontsize=10)
    ax.text(.025, .54, '120 Hz：接触与物理回调\n≤30 Hz：履带显示\n≤20 Hz：土壤网格显示', va='center', fontsize=11)
    ax.text(.025, .12, 'SCM 正常工作时：\n旧地面碰撞关闭；旧四区模型输入零载荷。\n带速由指令决定，没有负载转矩反馈。', fontsize=10, color=GRAY)
    ax.set_title('每步计算依赖关系：法向压力还会影响切向强度上限', pad=18)
    save(fig, '03_dynamics_pipeline')


def sampling():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    ax = axes[0]
    for x in range(-40, 41, 20): ax.axvline(x, color='#cbd3dc', lw=.8)
    for y in range(-40, 41, 20): ax.axhline(y, color='#cbd3dc', lw=.8)
    ax.add_patch(Rectangle((-45, -45), 45, 90, fc='#d0d9e2', ec=GRAY, alpha=.75))
    ax.add_patch(Rectangle((-10, -10), 20, 20, fc='none', ec=BLUE, lw=2))
    for x in (-5, 5):
        for y in (-5, 5):
            ax.plot(x, y, 'o' if x<0 else 'x', color=GREEN if x<0 else RED, ms=8, mew=2)
    dimension(ax, (-10, -20), (10, -20), '20 mm')
    ax.text(-42, 27, '履带板投影', fontsize=10)
    ax.text(0, -35, '2 / 4 命中：A = 0.0002 m²', ha='center', fontsize=10, color=BLUE)
    ax.set(xlim=(-48, 40), ylim=(-43, 44), xlabel=r'$x_W$ / mm', ylabel=r'$y_W$ / mm',
           title='(a) 边界单元示例：子采样偏置 ±Δ/4')
    ax.set_aspect('equal')
    ax = axes[1]
    ground_x = [-45, -24, -22.5, -6, -6, 6, 6, 22.5, 24, 45]
    ground_z = [0, 0, -12, -12, -20, -20, -12, -12, 0, 0]
    ax.fill_between(ground_x, ground_z, -40, color='#ebdfc9', hatch='...', edgecolor='#d9ccb6', lw=.5)
    ax.plot(ground_x, ground_z, color=ORANGE, lw=1.5)
    ax.add_patch(Rectangle((-22.5, -12), 45, 14, fc='#d0d9e2', ec=GRAY))
    ax.add_patch(Rectangle((-6, -20), 12, 8, fc=ORANGE, ec=ORANGE))
    ax.axhline(0, color=GRAY, ls='--', lw=1)
    for x, hit in ((-15, -12), (0, -20), (15, -12)):
        arrow(ax, (x, -37), (x, hit-1), color=BLUE)
        ax.plot(x, hit, 'o', color=RED, ms=4)
    dimension(ax, (34, -20), (34, 0), 'δ = 20 mm', offset=(10, 0))
    ax.text(-42, 5, '初始地表 $h_0$ = 0', fontsize=10)
    ax.text(-42, -31, '竖直直线 → 取最低进入交点', fontsize=9, color=BLUE)
    ax.set(xlim=(-47, 58), ylim=(-42, 14), xlabel=r'$x_W$ / mm', ylabel=r'$z_W$ / mm',
           title='(b) 履齿比板底低 8 mm：示例车体高度 0.20 m')
    ax.set_aspect('equal')
    fig.text(.5, .025, '四条射线先平均高度、位置与材料速度，再计算一个单元压力；不是四个独立土柱。两幅示例并非同一单元。', ha='center', fontsize=10)
    fig.subplots_adjust(bottom=.20, wspace=.30)
    save(fig, '04_contact_sampling')


def hysteresis():
    soil = SOIL['soil_parameters']['DrySand']
    ke, kphi, n = soil['elastic_pa_per_m'], soil['kphi'], soil['n']
    z = np.linspace(0, .03, 301)
    p = np.minimum(ke*z, kphi*z**n)
    zp = .03-p[-1]/ke
    unload_z = np.linspace(.03, 0, 301)
    unload_p = np.maximum(0, np.minimum(ke*(unload_z-zp), kphi*unload_z**n))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={'width_ratios': [1.3, 1]})
    ax = axes[0]
    ax.plot(z*1000, p/1000, lw=2.2, color=BLUE, label='新土准静态加载')
    ax.plot(unload_z*1000, unload_p/1000, lw=2.2, color=ORANGE, label='从 30 mm 卸载 / 同路径重载')
    ax.plot(zp*1000, 0, 'o', color=RED)
    ax.annotate(f'残余塑性沉陷 {zp*1000:.2f} mm', xy=(zp*1000, 0), xytext=(6, .7),
                fontsize=10, arrowprops={'arrowstyle': '->', 'color': RED}, color=RED)
    ax.set(xlabel=r'总压沉 $\delta$ / mm', ylabel=r'静态压力 $p_s$ / kPa',
           title='(a) 代码公式的准静态加载—卸载示例')
    ax.grid(alpha=.2)
    ax.legend(loc='upper left', fontsize=10)
    ax = axes[1]
    ax.set(xlim=(-.2, 2.8), ylim=(-37, 9))
    for x, label, top in ((.4, '受载', -30.), (1.9, '卸载', -zp*1000)):
        ax.add_patch(Rectangle((x-.35, -36), .7, top+36, fc='#e5d8be', ec=ORANGE))
        ax.plot([x-.42, x+.42], [top, top], color=ORANGE, lw=2)
        ax.text(x, -34, label, ha='center', fontsize=10)
    ax.axhline(0, color=GRAY, ls='--')
    ax.text(-.1, 2, '初始地表 $h_0$', fontsize=10)
    dimension(ax, (.93, -30), (.93, -zp*1000), '$z_e$', offset=(.18, 0))
    dimension(ax, (2.4, -zp*1000), (2.4, 0), '$z_p$', offset=(.18, 0))
    ax.text(1.15, -9, 'δ = $z_p$ + $z_e$', ha='center', fontsize=12)
    ax.text(1.15, -15, '卸载仅恢复弹性部分', ha='center', fontsize=10)
    ax.set(xticks=[], ylabel=r'地表高度 / mm', title='(b) 永久车辙与弹性恢复')
    ax.spines['bottom'].set_visible(False)
    fig.text(.5, .02, '按 DrySand 当前参数计算；关闭速度阻尼、规定压沉历程。曲线不是车辆实测或完整运行结果。', ha='center', fontsize=10)
    fig.subplots_adjust(bottom=.20, wspace=.33)
    save(fig, '05_normal_hysteresis')


def response_curves():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    z = np.linspace(0, .05, 200)
    colors = [GREEN, ORANGE, BLUE, '#8a597c', GRAY]
    ax = axes[0]
    for (name, p), color in zip(SOIL['soil_parameters'].items(), colors):
        ax.plot(z*1000, (p['kc']/.16+p['kphi'])*z**p['n']/1000, label=name, lw=1.8, color=color)
    ax.set(xlabel=r'$\delta$ / mm', ylabel=r'$p_B$ / kPa', title='(a) 五材料加载包络：配置值')
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    ax = axes[1]
    j = np.linspace(0, 5, 300)
    ax.plot(j, 1-np.exp(-j), color=BLUE, lw=2)
    ax.axhline(1, color=GRAY, ls='--', lw=1)
    ax.plot(1, 1-np.exp(-1), 'o', color=ORANGE)
    ax.text(1.15, .58, 'j = K：63.2%', fontsize=10, color=ORANGE)
    ax.text(2.2, .28, '本项目向量记忆限幅：|j| ≤ 5K\n此图未叠加切向速度阻尼', fontsize=9)
    ax.set(xlabel=r'$|\mathbf{j}|/K$', ylabel=r'$|\tau_{JH}|/\tau_{max}$', ylim=(0, 1.08),
           title='(b) Janosi 指数增长项')
    ax.grid(alpha=.2)
    ax = axes[2]
    t = np.arange(121)/120
    alpha = 1-np.exp(-(1/120)/.15)
    u = np.zeros_like(t)
    for k in range(1, len(t)): u[k] = u[k-1]+alpha*(.45-u[k-1])
    ax.axhline(.45, color=GRAY, ls='--', lw=1, label='速度指令')
    ax.plot(t, u, color=BLUE, lw=2, label='滤波带速')
    ax.axvline(.15, color=ORANGE, ls=':', lw=1)
    ax.text(.23, .16, '$T_m$ = 0.15 s\n无负载转矩反馈', fontsize=10)
    ax.set(xlabel='时间 / s', ylabel='带速 / (m/s)', title='(c) 120 Hz 简化驱动阶跃响应')
    ax.grid(alpha=.2)
    ax.legend(loc='lower right', fontsize=9)
    fig.text(.5, .015, '全部为当前公式与参数的解析 / 离散计算图，不代表实物标定结果。', ha='center', fontsize=10)
    fig.subplots_adjust(bottom=.20, wspace=.36)
    save(fig, '07_parameter_response')


def forces():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    ax = axes[0]
    ax.add_patch(Rectangle((-.5, .07), 1, .36, fc='#eaf0f5', ec=GRAY))
    ax.plot(0, .2, 'o', color=RED)
    ax.text(.03, .24, 'COM', color=RED)
    contact = (-.28, -.02)
    ax.plot(*contact, 'o', color=ORANGE)
    arrow(ax, contact, (-.28, .14), GREEN)
    arrow(ax, contact, (-.02, -.02), BLUE)
    arrow(ax, (0, .2), (0, .035), RED)
    arrow(ax, (0, .2), contact, GRAY)
    ax.text(-.40, .10, r'$p_iA_i$', color=GREEN)
    ax.text(-.13, -.07, r'$\tau_{x,i}A_i$', color=BLUE)
    ax.text(.025, .10, r'$mg$', color=RED)
    ax.text(-.22, .20, r'$\mathbf{r}_i$', color=GRAY)
    ax.axhline(0, color=ORANGE, lw=.8)
    ax.text(0, -.20, r'$\mathbf{F}=\sum_i \mathbf{f}_i,\quad \mathbf{M}=\sum_i\mathbf{r}_i\times\mathbf{f}_i$', ha='center', fontsize=12)
    ax.set(xlim=(-.70, .70), ylim=(-.27, .56), xlabel=r'$x_W$ / m', ylabel=r'$z_W$ / m',
           title='(a) 单元力对整车形成合力与力矩')
    ax.set_aspect('equal')
    ax = axes[1]
    for y in (-.35, .35): ax.add_patch(Rectangle((-.53, y-.08), 1.06, .16, fc='#e1e9ef', ec=GRAY))
    arrow(ax, (.15, .35), (-.20, .35), GREEN)
    arrow(ax, (-.15, -.35), (.20, -.35), GREEN)
    ax.text(0, .49, r'左：$u_L<0,\ F_{x,L}<0$', ha='center', fontsize=10, color=GREEN)
    ax.text(0, -.52, r'右：$u_R>0,\ F_{x,R}>0$', ha='center', fontsize=10, color=GREEN)
    for x, sign in ((.30, -1), (-.30, 1)):
        arrow(ax, (x, 0), (x, sign*.18), ORANGE)
    ax.text(.48, .0, '前部\n横向阻力', fontsize=9, ha='left', color=ORANGE)
    ax.text(-.49, -.15, '后部\n横向阻力', fontsize=9, ha='right', color=ORANGE)
    theta = np.linspace(-.4, 3.7, 80)
    ax.plot(.12*np.cos(theta), .12*np.sin(theta), color=BLUE, lw=1.5)
    arrow(ax, (.12*np.cos(theta[-4]), .12*np.sin(theta[-4])),
          (.12*np.cos(theta[-1]), .12*np.sin(theta[-1])), BLUE)
    ax.text(.0, -.23, r'$\Omega_z>0$', ha='center', fontsize=10, color=BLUE)
    ax.text(.0, .67, r'$M_{z,drive}\approx\frac{d}{2}(F_{x,R}-F_{x,L})>0$', ha='center', fontsize=12)
    ax.set(xlim=(-.80, .80), ylim=(-.66, .78), xlabel=r'$+x_B$ 向前 / m', ylabel=r'$+y_B$ 向左 / m',
           title='(b) 差速对转：牵引转矩与横向擦滑阻力')
    ax.set_aspect('equal')
    fig.text(.5, .025, '力箭头只表达方向和力臂，长度不代表计算数值。土壤支承沿世界 +Z；侧壁被动土压力未建模。', ha='center', fontsize=10)
    fig.subplots_adjust(bottom=.18, wspace=.28)
    save(fig, '06_force_and_turning')


def snapshot():
    files = ['configs/robot_v1.json', 'configs/tracks.json', 'configs/scm.json', 'configs/five_surfaces.json',
             'scripts/track_motion.py', 'scripts/track_visuals.py', 'scripts/scm_terrain.py',
             'scripts/scm_scene.py', 'scripts/amphibian_dynamics.py', 'scripts/robot_runtime.py',
             'scripts/ground_tools.py', 'scripts/ground_telemetry.py']
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, capture_output=True, text=True)
    soil = SOIL['soil_parameters']['DrySand']
    delta = .02
    static = (soil['kc']/.16+soil['kphi'])*delta**soil['n']
    pressure = static-soil['damping_pa_s_per_m']*(-.01)
    limit = soil['cohesion_pa']+pressure*np.tan(np.radians(soil['friction_deg']))
    j = .1/120
    tau_jh = limit*(-np.expm1(-j/soil['janosi_m']))
    tau = min(limit, tau_jh+soil['shear_damping_pa_s_per_m']*.1)
    data = {'documentation_date': '2026-10-31', 'base_commit': revision.stdout.strip(),
            'files_sha256': {f: hashlib.sha256((PROJECT/f).read_bytes()).hexdigest() for f in files},
            'geometry': {'radius_m': MOTION.radius, 'half_straight_m': MOTION.half_straight,
                         'perimeter_m': MOTION.perimeter, 'pitch_m': MOTION.pitch,
                         'straight_gap_m': MOTION.pitch-TRACK['shoe_length_m']},
            'dry_sand_single_cell_example': {'delta_m': delta, 'previous_plastic_m': 0,
                'p_static_pa': static, 'plastic_m': delta-static/soil['elastic_pa_per_m'],
                'elastic_m': static/soil['elastic_pa_per_m'], 'normal_velocity_m_s': -.01,
                'pressure_pa': pressure, 'relative_velocity_xy_m_s': [-.1, 0],
                'shear_m': [-j, 0], 'tau_max_pa': float(limit), 'tau_jh_pa': float(tau_jh),
                'tau_final_pa': float(tau), 'force_xy_z_n': [float(tau*.0004), 0, float(pressure*.0004)]},
            'figure_note': 'Geometry and formula illustrations; not native runtime screenshots or measured validation.'}
    (OUTPUT/'track_soil_snapshot.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    for function in (geometry, component, pipeline, sampling, hysteresis, response_curves, forces):
        function()
    snapshot()
    print('Generated seven PNG/SVG figure pairs and the source/config snapshot.')
