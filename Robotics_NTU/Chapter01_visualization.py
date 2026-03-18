import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).resolve().parent / 'images'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def Rx(deg):
    t = np.radians(deg)
    return np.array([[1, 0, 0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t),  np.cos(t)]])

def Ry(deg):
    t = np.radians(deg)
    return np.array([[ np.cos(t), 0, np.sin(t)],
                     [0,          1, 0],
                     [-np.sin(t), 0, np.cos(t)]])

def Rz(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0,          0,         1]])


def draw_frame(ax, R, origin=np.zeros(3), scale=1.0, label='', alpha=1.0):
    for i, c in enumerate(['r', 'g', 'b']):
        vec = R[:, i] * scale
        ax.quiver(*origin, *vec, color=c, alpha=alpha,
                  arrow_length_ratio=0.15, linewidth=2)
        ax.text(*(origin + vec * 1.1), f'{label}{"xyz"[i]}', color=c, fontsize=9)


def verify(R, name):
    print(f"\n{name}:")
    print(f"  R =\n{np.round(R, 4)}")
    print(f"  R^T R = I: {np.allclose(R.T @ R, np.eye(3))}")
    print(f"  det(R)   = {np.round(np.linalg.det(R), 6)}")


def plot_basic_rotations(angle=45):
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'三个基本旋转矩阵（旋转角 = {angle}°）', fontsize=14, fontweight='bold')

    cases = [
        (Rx(angle), f'$R_x({angle}°)$ 绕X轴'),
        (Ry(angle), f'$R_y({angle}°)$ 绕Y轴'),
        (Rz(angle), f'$R_z({angle}°)$ 绕Z轴'),
    ]

    for idx, (R, title) in enumerate(cases):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.set_title(title, pad=10)
        draw_frame(ax, np.eye(3), scale=0.8, alpha=0.2)
        draw_frame(ax, R, scale=0.8)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        verify(R, f'R_{"xyz"[idx]}({angle}°)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ch01_basic_rotations.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_non_commutativity():
    R1 = Rx(90) @ Rz(90)
    R2 = Rz(90) @ Rx(90)

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('旋转不可交换：$R_x(90°)R_z(90°) \\neq R_z(90°)R_x(90°)$',
                 fontsize=14, fontweight='bold')

    cases = [
        (R1, '$R_x R_z$：先绕Z，再绕X'),
        (R2, '$R_z R_x$：先绕X，再绕Z'),
    ]
    for idx, (R, title) in enumerate(cases):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        ax.set_title(title, pad=10)
        draw_frame(ax, np.eye(3), scale=0.7, alpha=0.2)
        draw_frame(ax, R, scale=0.7)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])

    print(f"\nR_x R_z =\n{np.round(R1, 4)}")
    print(f"\nR_z R_x =\n{np.round(R2, 4)}")
    print(f"\n相等: {np.allclose(R1, R2)}")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ch01_non_commutativity.png', dpi=150, bbox_inches='tight')
    plt.show()


def animate_rotation():
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    p0 = np.array([1.0, 0.0, 0.5])
    angles = np.linspace(0, 360, 100)
    traj = np.array([Rz(a) @ p0 for a in angles])

    ax.plot(*traj.T, 'b--', alpha=0.3, label='轨迹')
    ax.scatter(*p0, color='green', s=100, label='初始点', zorder=5)
    point, = ax.plot([p0[0]], [p0[1]], [p0[2]], 'ro', markersize=10)
    draw_frame(ax, np.eye(3), scale=0.5)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-0.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    def update(frame):
        p = traj[frame]
        point.set_data([p[0]], [p[1]])
        point.set_3d_properties([p[2]])
        ax.set_title(f'绕Z轴旋转，角度 = {angles[frame]:.0f}°')
        return point,

    anim = FuncAnimation(fig, update, frames=len(angles), interval=30, blit=False)
    plt.tight_layout()
    plt.show()
    return anim


if __name__ == '__main__':
    plot_basic_rotations(angle=45)
    plot_non_commutativity()
    # animate_rotation()
