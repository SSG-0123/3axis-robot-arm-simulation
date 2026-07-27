#!/usr/bin/env python3
"""
robot_arm_3d_simulation.py
==========================
3D空間 3軸ロボットアーム ボール把持シミュレーション
3D Space 3-Axis (6-DOF) Robot Arm Ball Grasping Simulation

Author : SSG-0123
Date   : 2026

Description
-----------
デカルトコンヘ回転に基づく DHパラメータ記述（簡略化モデル）を使用し、
6自由度（J1～J6）のロボットアームがボールを把持するシミュレーション。

機能
----
- 3D アニメーション  (matplotlib Axes3D)
- 順運動学 (FK)  — 廂数積 DH変換行列
- 逆運動学 (IK)  — 3D ヤコビアン擬似逆行列法
- 状態機械による把持動作計画
- エンドエフェクタ 3D 軌跡表示
- グリッパー描画（間口開閉完全对応）
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time

# ============================================================
# 定数
# ============================================================
DEG = np.pi / 180

# ============================================================
# データ構造
# ============================================================

@dataclass
class JointState:
    """6軸分の関節角度 [rad]"""
    q: np.ndarray = field(default_factory=lambda: np.array([
        0.0,          # J1 : ベース回転 (Z軸)
        45 * DEG,     # J2 : 肩仰き (Y軸)
       -60 * DEG,     # J3 : 肘 (Y軸)
        0.0,          # J4 : 前腕回転 (Z軸)
        30 * DEG,     # J5 : 手首 (Y軸)
        0.0,          # J6 : 手首回転 (Z軸)
    ]))

    def copy(self) -> 'JointState':
        return JointState(q=self.q.copy())


@dataclass
class Ball3D:
    """3Dボールの状態"""
    pos: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.6, 0.15]))
    radius: float   = 0.08
    color: str      = 'orange'
    is_grasped: bool = False


# ============================================================
# 3Dロボットアーム (6DOF 簡略化 DHモデル)
# ============================================================

class RobotArm6DOF:
    """
    6自由度ロボットアーム。

    DHパラメータ記述 (簡略化: alpha=0, d=0 以外は辺長 a のみ):
      J1: ベース回転  a=0,    d=d1
      J2: 肩仰き    a=a2,   d=0
      J3: 肘          a=a3,   d=0
      J4: 前腕回転  a=0,    d=d4
      J5: 手首仰き    a=0,    d=0
      J6: ツール回転  a=0,    d=d6 (エンドエフェクタ長)
    """

    # リンク寘法
    D1  = 0.20   # ベース高さ
    A2  = 0.50   # 上腕長
    A3  = 0.40   # 前腕長
    D4  = 0.10   # 前腕オフセット
    D6  = 0.12   # EE長

    JOINT_LIMITS = np.array([
        [-180, 180],   # J1
        [ -90, 135],   # J2
        [-170,  10],   # J3
        [-360, 360],   # J4
        [-135, 135],   # J5
        [-360, 360],   # J6
    ]) * DEG

    def __init__(self):
        self.state = JointState()
        self.gripper_open  = True
        self.gripper_width = 0.06

    # ----------------------------------------------------------
    # 単一 DH 変換行列
    # ----------------------------------------------------------
    @staticmethod
    def _dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """標準 Denavit–Hartenberg 変換行列 (4x4)"""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct,  -st*ca,  st*sa,  a*ct],
            [st,   ct*ca, -ct*sa,  a*st],
            [0,      sa,    ca,     d  ],
            [0,       0,     0,     1  ],
        ])

    # ----------------------------------------------------------
    # 順運動学
    # ----------------------------------------------------------
    def forward_kinematics(self,
                           q: Optional[np.ndarray] = None
                           ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        FK 計算。
        Returns:
            joint_positions : 各関節の 3D 座標リスト (7点: base + J1〜EE)
            T_ee            : エンドエフェクタの変換行列 (4x4)
        """
        if q is None:
            q = self.state.q

        q1, q2, q3, q4, q5, q6 = q

        # DHパラメータ: (theta, d, a, alpha)
        dh_params = [
            (q1,  self.D1,  0,        90*DEG),
            (q2,  0,        self.A2,  0     ),
            (q3,  0,        self.A3,  0     ),
            (q4,  self.D4,  0,        90*DEG),
            (q5,  0,        0,       -90*DEG),
            (q6,  self.D6,  0,        0     ),
        ]

        T = np.eye(4)
        positions = [T[:3, 3].copy()]   # base = origin

        for params in dh_params:
            T = T @ self._dh_matrix(*params)
            positions.append(T[:3, 3].copy())

        return positions, T

    # ----------------------------------------------------------
    # 3Dヤコビアン (6x3)
    # ----------------------------------------------------------
    def _jacobian_3d(self, q: np.ndarray) -> np.ndarray:
        """数値微分による 3D エンドエフェクタ位置ヤコビアン (3x6)"""
        delta = 1e-6
        _, T0 = self.forward_kinematics(q)
        ee0   = T0[:3, 3]
        J     = np.zeros((3, 6))

        for i in range(6):
            dq       = q.copy()
            dq[i]   += delta
            _, T1    = self.forward_kinematics(dq)
            ee1      = T1[:3, 3]
            J[:, i]  = (ee1 - ee0) / delta

        return J

    # ----------------------------------------------------------
    # 逆運動学 (Jacobian Pseudo-Inverse + Damped Least Squares)
    # ----------------------------------------------------------
    def inverse_kinematics(self,
                            target: np.ndarray,
                            max_iter: int = 600,
                            tol: float    = 1e-4,
                            alpha: float  = 0.8,
                            lambda_: float= 0.01
                            ) -> bool:
        """
        Damped Least Squares IK (期所波内行列法)。
        J^T (J J^T + λ^2 I)^{-1} エラーベクトルで関節角度を更新。
        """
        q = self.state.q.copy()

        for _ in range(max_iter):
            _, T = self.forward_kinematics(q)
            ee   = T[:3, 3]
            err  = target - ee

            if np.linalg.norm(err) < tol:
                self.state.q = np.clip(q, self.JOINT_LIMITS[:, 0], self.JOINT_LIMITS[:, 1])
                return True

            J     = self._jacobian_3d(q)
            # Damped Least Squares
            JJT   = J @ J.T
            dq    = alpha * J.T @ np.linalg.solve(
                        JJT + (lambda_**2) * np.eye(3), err)

            q = np.clip(q + dq,
                        self.JOINT_LIMITS[:, 0],
                        self.JOINT_LIMITS[:, 1])

        self.state.q = q
        return False

    # ----------------------------------------------------------
    # 到達可能範囲チェック
    # ----------------------------------------------------------
    def max_reach(self) -> float:
        return self.A2 + self.A3 + self.D4 + self.D6

    def is_reachable(self, pos: np.ndarray) -> bool:
        dist = np.linalg.norm(pos[:2])  # XY平面距離
        return dist <= self.max_reach() * 0.97


# ============================================================
# 状態機械
# ============================================================

class SimState:
    IDLE      = 'IDLE'
    APPROACH  = 'APPROACH'   # 上空にアプローチ
    DESCEND   = 'DESCEND'    # 把持位置へ降下
    GRASP     = 'GRASP'      # 把持
    LIFT      = 'LIFT'       # 持ち上げ
    TRANSIT   = 'TRANSIT'    # 配置先へ移動
    PLACE     = 'PLACE'      # 配置位置へ降下
    RELEASE   = 'RELEASE'    # 解放
    RETRACT   = 'RETRACT'    # 引こ指
    RETURN    = 'RETURN'     # ホームへ
    DONE      = 'DONE'


# ============================================================
# メインシミュレーション
# ============================================================

class BallGraspSim3D:
    """
    3Dボール把持シミュレーション。
    matplotlib Axes3D でリアルタイムアニメーション。
    """

    APPROACH_OFFSET = 0.25   # 上空アプローチZオフセット [m]
    PLACE_POS = np.array([-0.6, 0.5, 0.15])  # 配置先座標

    def __init__(self):
        self.arm   = RobotArm6DOF()
        self.ball  = Ball3D()
        self.state = SimState.IDLE
        self.log_msgs: List[str] = []

        # ウェイポイントキュー (target_xyz)
        self.waypoints: List[np.ndarray] = []
        self.wp_idx    = 0
        self.wp_thresh = 0.018   # 到達判定閾値 [m]

        # 軌跡バッファ
        self.ee_traj: List[np.ndarray] = []

        # Figure セットアップ
        self.fig = plt.figure(figsize=(14, 9))
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self._plan_waypoints()
        self._log("3Dシミュレーション初期化完了")

    # ----------------------------------------------------------
    # ウェイポイント計画
    # ----------------------------------------------------------
    def _plan_waypoints(self):
        bx, by, bz = self.ball.pos
        px, py, pz = self.PLACE_POS
        ao = self.APPROACH_OFFSET

        self.waypoints = [
            np.array([bx, by, bz + ao]),          # 0: ボール上空
            np.array([bx, by, bz + self.arm.D6]), # 1: 把持座標
            np.array([bx, by, bz + ao]),          # 2: リフトアップ
            np.array([px, py, pz + ao + 0.1]),    # 3: 配置上空(ホープイント)
            np.array([px, py, pz + self.arm.D6]), # 4: 配置座標
            np.array([px, py, pz + ao]),          # 5: 引き上げ
            np.array([0.6, 0.0, 0.5]),            # 6: ホーム姿勢
        ]

        # 状態遷移マッピング (wp番号 -> 遷移先状態)
        self._wp_state_map = {
            0: SimState.APPROACH,
            1: SimState.DESCEND,
            2: SimState.GRASP,
            3: SimState.LIFT,
            4: SimState.TRANSIT,
            5: SimState.PLACE,
            6: SimState.RELEASE,
        }
        self._log(f"軌道計画: {len(self.waypoints)} WP")

    # ----------------------------------------------------------
    # ログ
    # ----------------------------------------------------------
    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.log_msgs.append(entry)
        print(entry)

    # ----------------------------------------------------------
    # 状態遷移
    # ----------------------------------------------------------
    def _on_waypoint_reached(self, wp_idx: int):
        state_map = {
            0: (SimState.DESCEND,  "上空到達 → 降下開始"),
            1: (SimState.GRASP,    "把持座標到達 → グリッパー閉"),
            2: (SimState.LIFT,     "把持完了 → リフト"),
            3: (SimState.TRANSIT,  "リフト完了 → 配置先へ"),
            4: (SimState.PLACE,    "配置上空到達 → 降下"),
            5: (SimState.RELEASE,  "配置位置到達 → 解放"),
            6: (SimState.RETRACT,  "解放完了 → 引き上げ"),
        }

        if wp_idx in state_map:
            new_state, msg = state_map[wp_idx]
            self.state = new_state
            self._log(msg)

            if wp_idx == 1:   # 把持
                self.arm.gripper_open  = False
                self.ball.is_grasped   = True
                self.ball.color        = 'crimson'
                self._log("  Gripper: CLOSED")

            elif wp_idx == 5:  # 解放
                self.arm.gripper_open  = True
                self.ball.is_grasped   = False
                self.ball.pos          = self.PLACE_POS.copy()
                self.ball.color        = 'limegreen'
                self._log("  Gripper: OPEN")

        elif wp_idx == len(self.waypoints) - 1:
            self.state = SimState.DONE
            self._log("シミュレーション完了!")

    # ----------------------------------------------------------
    # 3Dグリッパー描画ヘルパー
    # ----------------------------------------------------------
    def _draw_gripper(self, positions: List[np.ndarray]):
        ee = positions[-1]
        # EEの姿勢ベクトル (J5-EE 方向から垂直)
        dir_vec = ee - positions[-2]
        norm    = np.linalg.norm(dir_vec)
        if norm < 1e-8:
            dir_vec = np.array([0, 0, 1.0])
        else:
            dir_vec /= norm

        # 垂直方向
        perp = np.cross(dir_vec, np.array([0, 0, 1]))
        if np.linalg.norm(perp) < 1e-8:
            perp = np.cross(dir_vec, np.array([1, 0, 0]))
        perp /= np.linalg.norm(perp)

        gw = self.arm.gripper_width if self.arm.gripper_open else 0.015
        gc = 'dimgray' if self.arm.gripper_open else 'darkred'

        for sign in [+1, -1]:
            tip = ee + perp * sign * gw
            self.ax.plot([ee[0], tip[0]], [ee[1], tip[1]], [ee[2], tip[2]],
                         color=gc, linewidth=4, zorder=5)
            # 指先クロー
            tip2 = tip + dir_vec * 0.04
            self.ax.plot([tip[0], tip2[0]], [tip[1], tip2[1]], [tip[2], tip2[2]],
                         color=gc, linewidth=4, zorder=5)

    # ----------------------------------------------------------
    # 3Dボール描画 (Wireframe球)
    # ----------------------------------------------------------
    def _draw_ball(self):
        r = self.ball.radius
        cx, cy, cz = self.ball.pos
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 12)
        x = cx + r * np.outer(np.cos(u), np.sin(v))
        y = cy + r * np.outer(np.sin(u), np.sin(v))
        z = cz + r * np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_surface(x, y, z, color=self.ball.color,
                              alpha=0.85, zorder=4)

    # ----------------------------------------------------------
    # アニメーションフレーム更新
    # ----------------------------------------------------------
    def update(self, frame: int):
        self.ax.cla()
        self._setup_axes()

        # --- 状態機械ステップ ---
        if self.state == SimState.IDLE:
            self.state = SimState.APPROACH
            self._log("シミュレーション開始")

        if self.state != SimState.DONE and self.wp_idx < len(self.waypoints):
            wp   = self.waypoints[self.wp_idx]
            ok   = self.arm.inverse_kinematics(wp)
            pos_list, _ = self.arm.forward_kinematics()
            ee   = pos_list[-1]

            # 軌跡記録
            self.ee_traj.append(ee.copy())

            # 把持中はボールがEEに追従
            if self.ball.is_grasped:
                self.ball.pos = ee - np.array([0, 0, self.arm.D6])

            # 到達判定
            if np.linalg.norm(ee - wp) < self.wp_thresh:
                self._on_waypoint_reached(self.wp_idx)
                self.wp_idx += 1

        else:
            pos_list, _ = self.arm.forward_kinematics()

        # --- 描画 ---
        self._draw_arm(pos_list)
        self._draw_gripper(pos_list)
        self._draw_ball()
        self._draw_place_target()
        self._draw_trajectory()
        self._draw_workspace()
        self._draw_waypoints()
        self._draw_ui_text(frame, pos_list[-1])

    # ----------------------------------------------------------
    # アーム描画
    # ----------------------------------------------------------
    def _draw_arm(self, positions: List[np.ndarray]):
        link_colors = ['#1565C0', '#1565C0', '#2E7D32',
                        '#2E7D32', '#E65100', '#E65100']
        link_labels = ['Link1', 'Link2', 'Link3', 'Link4', 'Link5', 'Link6']
        link_widths = [8, 7, 6, 5, 4, 4]

        for i in range(len(positions) - 1):
            p0, p1 = positions[i], positions[i+1]
            self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                         color=link_colors[i], linewidth=link_widths[i],
                         solid_capstyle='round',
                         label=link_labels[i] if i < 3 else None)

        # 関節マーカー
        joint_names = ['Base', 'J1', 'J2', 'J3', 'J4', 'J5', 'EE']
        joint_colors = ['#212121', '#1565C0', '#1565C0',
                         '#2E7D32', '#2E7D32', '#E65100', '#BF360C']
        sizes = [60, 40, 40, 40, 30, 30, 50]

        for pt, name, col, sz in zip(positions, joint_names,
                                      joint_colors, sizes):
            self.ax.scatter(*pt, color=col, s=sz, zorder=6,
                             depthshade=False)

    # ----------------------------------------------------------
    # 軌跡描画
    # ----------------------------------------------------------
    def _draw_trajectory(self):
        if len(self.ee_traj) < 2:
            return
        traj = np.array(self.ee_traj)
        self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                     'c--', linewidth=1.5, alpha=0.55,
                     label='EE Trajectory')

    # ----------------------------------------------------------
    # 配置ターゲット描画
    # ----------------------------------------------------------
    def _draw_place_target(self):
        px, py, pz = self.PLACE_POS
        # 円リング
        theta = np.linspace(0, 2*np.pi, 40)
        r = self.ball.radius * 1.5
        self.ax.plot(px + r*np.cos(theta), py + r*np.sin(theta),
                     [pz]*40, color='purple', linewidth=2,
                     linestyle='--', label='Place Target')
        self.ax.scatter(px, py, pz, color='purple', s=80,
                         marker='*', depthshade=False, zorder=7)

    # ----------------------------------------------------------
    # 到達可能内益描画
    # ----------------------------------------------------------
    def _draw_workspace(self):
        r = self.arm.max_reach()
        theta = np.linspace(0, 2*np.pi, 60)
        phi   = np.linspace(0,   np.pi, 30)
        # XY平面円
        self.ax.plot(r*np.cos(theta), r*np.sin(theta),
                     [self.arm.D1]*60, color='gray',
                     linewidth=0.7, alpha=0.25, linestyle=':')

    # ----------------------------------------------------------
    # ウェイポイント表示
    # ----------------------------------------------------------
    def _draw_waypoints(self):
        for i, wp in enumerate(self.waypoints):
            if i < self.wp_idx:
                m, c, s = 'v', 'gray',  20
            elif i == self.wp_idx:
                m, c, s = 'D', 'gold',  50
            else:
                m, c, s = '^', 'navy',  20
            self.ax.scatter(wp[0], wp[1], wp[2],
                             marker=m, color=c, s=s, depthshade=False)

    # ----------------------------------------------------------
    # 軌跡設定
    # ----------------------------------------------------------
    def _setup_axes(self):
        lim = 1.2
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(0, lim * 1.2)
        self.ax.set_xlabel('X [m]', labelpad=6)
        self.ax.set_ylabel('Y [m]', labelpad=6)
        self.ax.set_zlabel('Z [m]', labelpad=6)
        self.ax.set_title('3D Robot Arm — Ball Grasping Simulation',
                           fontsize=13, fontweight='bold', pad=12)
        # 地面グリッド
        xx, yy = np.meshgrid(np.linspace(-lim, lim, 6),
                              np.linspace(-lim, lim, 6))
        self.ax.plot_surface(xx, yy, np.zeros_like(xx),
                              alpha=0.08, color='tan')
        self.ax.view_init(elev=22, azim=225)

    # ----------------------------------------------------------
    # UIテキスト
    # ----------------------------------------------------------
    def _draw_ui_text(self, frame: int, ee_pos: np.ndarray):
        q_deg = np.degrees(self.arm.state.q)
        grip  = "OPEN" if self.arm.gripper_open else "CLOSED"

        state_colors = {
            SimState.IDLE:     'gray',
            SimState.APPROACH: 'dodgerblue',
            SimState.DESCEND:  'royalblue',
            SimState.GRASP:    'tomato',
            SimState.LIFT:     'orange',
            SimState.TRANSIT:  'darkorchid',
            SimState.PLACE:    'teal',
            SimState.RELEASE:  'green',
            SimState.RETRACT:  'olive',
            SimState.RETURN:   'navy',
            SimState.DONE:     'limegreen',
        }
        sc = state_colors.get(self.state, 'black')

        info = (
            f"State : {self.state}\n"
            f"Frame : {frame}\n"
            f"WP    : {self.wp_idx}/{len(self.waypoints)}\n"
            f"Gripper: {grip}\n"
            f"EE pos: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})\n"
            f"---Joint Angles [deg]---\n"
            f"J1:{q_deg[0]:+6.1f}  J4:{q_deg[3]:+6.1f}\n"
            f"J2:{q_deg[1]:+6.1f}  J5:{q_deg[4]:+6.1f}\n"
            f"J3:{q_deg[2]:+6.1f}  J6:{q_deg[5]:+6.1f}"
        )

        self.ax.text2D(0.01, 0.99, info,
                        transform=self.ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='white',
                                  alpha=0.8, edgecolor=sc, linewidth=2))

        if self.state == SimState.DONE:
            self.ax.text2D(0.5, 0.5, 'GRASP\nCOMPLETE!',
                            transform=self.ax.transAxes,
                            fontsize=26, color='limegreen',
                            fontweight='bold', ha='center', va='center',
                            alpha=0.45)

        self.ax.legend(loc='upper right', fontsize=7,
                        framealpha=0.8, ncol=1)

    # ----------------------------------------------------------
    # 実行
    # ----------------------------------------------------------
    def run(self, frames: int = 250, interval: int = 70,
             save_gif: bool = False, gif_path: str = 'robot_arm_3d.gif'):
        """
        シミュレーション実行エントリポイント。

        Args:
            frames    : 総フレーム数
            interval  : ms/frame
            save_gif  : True なら GIF 保存
            gif_path  : GIF 保存パス
        """
        ani = animation.FuncAnimation(
            self.fig, self.update,
            frames=frames, interval=interval,
            repeat=False
        )

        if save_gif:
            self._log(f"GIF保存中: {gif_path}")
            ani.save(gif_path, writer='pillow', fps=12,
                      dpi=90, savefig_kwargs={'facecolor': 'white'})
            self._log("保存完了")
        else:
            plt.tight_layout()
            plt.show()


# ============================================================
# エントリポイント
# ============================================================

if __name__ == '__main__':
    print("=" * 65)
    print(" 3Dロボットアーム 6DOF ボール把持シミュレーション")
    print(" 3D Robot Arm (6DOF) Ball Grasping Simulation")
    print("=" * 65)
    print()
    print("アーム仕様:")
    print(f"  Arm type    : 6DOF (J1〜J6)")
    print(f"  Base height : {RobotArm6DOF.D1*100:.0f} cm")
    print(f"  Upper arm   : {RobotArm6DOF.A2*100:.0f} cm")
    print(f"  Forearm     : {RobotArm6DOF.A3*100:.0f} cm")
    print(f"  Max reach   : {RobotArm6DOF.A2+RobotArm6DOF.A3+RobotArm6DOF.D4+RobotArm6DOF.D6:.2f} m")
    print()
    print("逆運動学: Damped Least Squares (J^T(JJ^T+λI)^{-1} e)")
    print()
    print("動作フロー:")
    print("  IDLE → APPROACH → DESCEND → GRASP")
    print("       → LIFT → TRANSIT → PLACE → RELEASE → DONE")
    print()

    sim = BallGraspSim3D()
    # GIF保存時は save_gif=True に変更
    sim.run(frames=250, interval=70, save_gif=False)
