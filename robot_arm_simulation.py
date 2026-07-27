#!/usr/bin/env python3
"""
3軸ロボットアーム ボール把持シミュレーション
3-Axis Robot Arm Ball Grasping Simulation

Author: SSG-0123
Description:
    3軸（肩・肘・手首）ロボットアームが
    ターゲットボールの位置を検出し、逆運動学（IK）を用いて
    軌道計画・把持動作をシミュレーションするプログラム。
    matplotlibによるリアルタイム2Dアニメーション付き。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

# ============================================================
# データ構造定義
# ============================================================

@dataclass
class JointAngles:
    """3軸ロボットアームの関節角度 [rad]"""
    theta1: float = 0.0  # 肩 (shoulder)
    theta2: float = 0.0  # 肘 (elbow)
    theta3: float = 0.0  # 手首 (wrist)

@dataclass
class Point3D:
    """3D空間上の座標"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Ball:
    """ボールの状態"""
    position: Point3D
    radius: float = 0.05
    color: str = 'orange'
    is_grasped: bool = False

# ============================================================
# ロボットアームクラス
# ============================================================

class RobotArm3Axis:
    """
    3軸ロボットアーム（2D平面上のシミュレーション）

    座標系:
        - ベース原点: (0, 0)
        - 正のX方向: 右
        - 正のY方向: 上

    リンク長:
        - L1: 肩-肘リンク
        - L2: 肘-手首リンク
        - L3: 手首-エンドエフェクタリンク
    """

    def __init__(self, L1: float = 1.0, L2: float = 0.8, L3: float = 0.4):
        self.L1 = L1  # 上腕リンク長
        self.L2 = L2  # 前腕リンク長
        self.L3 = L3  # 手首リンク長
        self.angles = JointAngles(theta1=np.pi/4, theta2=-np.pi/4, theta3=0.0)
        self.gripper_open = True
        self.gripper_width = 0.15  # グリッパー開放幅

        # 可動範囲 [rad]
        self.joint_limits = {
            'theta1': (-np.pi / 2, np.pi),
            'theta2': (-np.pi * 0.9, np.pi * 0.9),
            'theta3': (-np.pi, np.pi),
        }

    # ----------------------------------------------------------
    # 順運動学 (Forward Kinematics)
    # ----------------------------------------------------------
    def forward_kinematics(self, angles: Optional[JointAngles] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        各関節の絶対座標を計算して返す。
        Returns:
            base, joint1, joint2, end_effector の座標 (x, y)
        """
        if angles is None:
            angles = self.angles

        t1 = angles.theta1
        t2 = angles.theta2
        t3 = angles.theta3

        base = np.array([0.0, 0.0])

        # 関節1 (肩)
        j1 = base + np.array([self.L1 * np.cos(t1), self.L1 * np.sin(t1)])

        # 関節2 (肘)
        j2 = j1 + np.array([
            self.L2 * np.cos(t1 + t2),
            self.L2 * np.sin(t1 + t2)
        ])

        # エンドエフェクタ (手首)
        ee = j2 + np.array([
            self.L3 * np.cos(t1 + t2 + t3),
            self.L3 * np.sin(t1 + t2 + t3)
        ])

        return base, j1, j2, ee

    # ----------------------------------------------------------
    # 逆運動学 (Inverse Kinematics) - 数値解法（ヤコビアン法）
    # ----------------------------------------------------------
    def inverse_kinematics(self, target_x: float, target_y: float,
                           max_iter: int = 500, tol: float = 1e-4) -> bool:
        """
        ヤコビアン擬似逆行列法（Jacobian Pseudo-Inverse）による逆運動学。
        目標座標 (target_x, target_y) へエンドエフェクタを移動。

        Returns:
            bool: 収束成功フラグ
        """
        target = np.array([target_x, target_y])
        alpha = 0.5  # 学習率

        for _ in range(max_iter):
            _, _, _, ee = self.forward_kinematics()
            error = target - ee

            if np.linalg.norm(error) < tol:
                return True

            # ヤコビアン行列 J (2x3) を数値微分で計算
            J = np.zeros((2, 3))
            delta = 1e-5
            thetas = [self.angles.theta1, self.angles.theta2, self.angles.theta3]

            for i in range(3):
                d_angles = JointAngles(*thetas)
                perturbed = list(thetas)
                perturbed[i] += delta
                d_angles = JointAngles(*perturbed)
                _, _, _, ee_d = self.forward_kinematics(d_angles)
                J[:, i] = (ee_d - ee) / delta

            # 擬似逆行列
            J_pinv = np.linalg.pinv(J)
            delta_theta = alpha * J_pinv @ error

            # 角度更新 + 可動範囲クリップ
            new_t1 = np.clip(self.angles.theta1 + delta_theta[0],
                             *self.joint_limits['theta1'])
            new_t2 = np.clip(self.angles.theta2 + delta_theta[1],
                             *self.joint_limits['theta2'])
            new_t3 = np.clip(self.angles.theta3 + delta_theta[2],
                             *self.joint_limits['theta3'])
            self.angles = JointAngles(new_t1, new_t2, new_t3)

        return False

    # ----------------------------------------------------------
    # リンク長合計（到達可能範囲チェック）
    # ----------------------------------------------------------
    def max_reach(self) -> float:
        return self.L1 + self.L2 + self.L3

    def is_reachable(self, x: float, y: float) -> bool:
        dist = np.sqrt(x**2 + y**2)
        return dist <= self.max_reach() * 0.98

# ============================================================
# シミュレーション状態機械
# ============================================================

class SimulationState:
    """シミュレーションのフェーズを管理する状態機械"""
    IDLE         = 'IDLE'          # 待機
    MOVING_TO    = 'MOVING_TO'     # ボールへ移動中
    GRASPING     = 'GRASPING'      # 把持中
    LIFTING      = 'LIFTING'       # 持ち上げ中
    PLACING      = 'PLACING'       # 配置位置へ移動中
    RELEASING    = 'RELEASING'     # 解放中
    RETURNING    = 'RETURNING'     # 初期位置へ戻り中
    DONE         = 'DONE'          # 完了

# ============================================================
# メインシミュレーションクラス
# ============================================================

class BallGraspingSimulation:
    """
    3軸ロボットアームによるボール把持シミュレーション管理クラス。
    アニメーション描画・状態遷移・軌道計画を統括する。
    """

    def __init__(self):
        self.arm = RobotArm3Axis(L1=1.0, L2=0.8, L3=0.4)
        self.ball = Ball(position=Point3D(x=1.5, y=0.3, z=0.0), radius=0.10)
        self.place_target = Point3D(x=-1.2, y=0.5, z=0.0)  # 配置先
        self.state = SimulationState.IDLE
        self.state_timer = 0
        self.trajectory: List[np.ndarray] = []  # エンドエフェクタ軌跡
        self.status_log: List[str] = []
        self.approach_height = 0.6  # アプローチ高さオフセット

        # 目標経路キュー
        self.waypoints: List[Tuple[float, float]] = []
        self.current_wp_idx = 0

        # アニメーション用
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self._setup_plot()
        self._plan_trajectory()

    # ----------------------------------------------------------
    # 軌道計画
    # ----------------------------------------------------------
    def _plan_trajectory(self):
        """
        ウェイポイント（経由点）ベースの軌道計画:
        1. アーム真上アプローチ点
        2. ボール把持位置
        3. リフトアップ
        4. 配置位置上空
        5. 配置位置
        6. 初期姿勢
        """
        bx, by = self.ball.position.x, self.ball.position.y
        px, py = self.place_target.x, self.place_target.y

        self.waypoints = [
            (bx, by + self.approach_height),  # 1. 上空アプローチ
            (bx, by + self.ball.radius),       # 2. 把持位置
            (bx, by + self.approach_height),  # 3. リフトアップ
            (px, py + self.approach_height),  # 4. 配置上空
            (px, py + self.ball.radius),       # 5. 配置位置
            (px, py + self.approach_height),  # 6. 引き上げ
            (1.0, 1.0),                        # 7. 初期姿勢付近
        ]
        self.current_wp_idx = 0
        self.log(f"軌道計画完了: {len(self.waypoints)}ウェイポイント")

    # ----------------------------------------------------------
    # プロット初期化
    # ----------------------------------------------------------
    def _setup_plot(self):
        self.ax.set_xlim(-2.5, 3.0)
        self.ax.set_ylim(-0.3, 3.0)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X [m]', fontsize=12)
        self.ax.set_ylabel('Y [m]', fontsize=12)
        self.ax.set_title('3-Axis Robot Arm - Ball Grasping Simulation', fontsize=14, fontweight='bold')
        # 地面
        self.ax.axhline(y=0, color='saddlebrown', linewidth=3, label='Ground')
        self.ax.fill_between([-3, 4], [-0.3, -0.3], [0, 0], color='peru', alpha=0.3)

    # ----------------------------------------------------------
    # ログ
    # ----------------------------------------------------------
    def log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        self.status_log.append(full_msg)
        print(full_msg)

    # ----------------------------------------------------------
    # 状態遷移ロジック
    # ----------------------------------------------------------
    def _state_label_map(self) -> dict:
        return {
            SimulationState.IDLE:      "待機中 (IDLE)",
            SimulationState.MOVING_TO: f"移動中 → WP{self.current_wp_idx+1}/{len(self.waypoints)}",
            SimulationState.GRASPING:  "把持中 (GRASPING)",
            SimulationState.LIFTING:   "持ち上げ中 (LIFTING)",
            SimulationState.PLACING:   "配置移動中 (PLACING)",
            SimulationState.RELEASING: "解放中 (RELEASING)",
            SimulationState.RETURNING: "復帰中 (RETURNING)",
            SimulationState.DONE:      "完了! (DONE)",
        }

    def _advance_state(self):
        """ウェイポイントに到達したときの状態遷移"""
        wp_actions = {
            1: (SimulationState.MOVING_TO, "ボール上空到達 → 降下開始"),
            2: (SimulationState.GRASPING,  "把持位置到達 → グリッパー閉じる"),
            3: (SimulationState.LIFTING,   "把持完了 → リフトアップ"),
            4: (SimulationState.PLACING,   "配置上空到達 → 降下開始"),
            5: (SimulationState.RELEASING, "配置位置到達 → グリッパー開放"),
            6: (SimulationState.RETURNING, "解放完了 → 初期姿勢へ復帰"),
            7: (SimulationState.DONE,      "初期姿勢復帰完了"),
        }
        wp_num = self.current_wp_idx + 1
        if wp_num in wp_actions:
            new_state, msg = wp_actions[wp_num]
            self.state = new_state
            self.log(msg)

            # グリッパー操作
            if new_state == SimulationState.GRASPING:
                self.arm.gripper_open = False
                self.ball.is_grasped = True
                self.ball.color = 'red'
                self.log("  >> グリッパー: CLOSED (把持完了)")
            elif new_state == SimulationState.RELEASING:
                self.arm.gripper_open = True
                self.ball.is_grasped = False
                self.ball.position.x = self.place_target.x
                self.ball.position.y = self.place_target.y
                self.ball.color = 'green'
                self.log("  >> グリッパー: OPEN (解放完了)")

    # ----------------------------------------------------------
    # アニメーション更新
    # ----------------------------------------------------------
    def update(self, frame: int):
        self.ax.cla()
        self._setup_plot()

        # --- 状態機械ステップ ---
        if self.state == SimulationState.IDLE:
            self.state = SimulationState.MOVING_TO
            self.log("シミュレーション開始")

        if self.state != SimulationState.DONE and self.current_wp_idx < len(self.waypoints):
            wp = self.waypoints[self.current_wp_idx]
            success = self.arm.inverse_kinematics(wp[0], wp[1])
            _, _, _, ee = self.arm.forward_kinematics()
            dist = np.linalg.norm(ee - np.array(wp))

            # 軌跡記録
            self.trajectory.append(ee.copy())

            # ボールがEEに追従（把持中）
            if self.ball.is_grasped:
                self.ball.position.x = ee[0]
                self.ball.position.y = ee[1] - self.ball.radius

            # ウェイポイント到達判定
            if dist < 0.05:
                self._advance_state()
                self.current_wp_idx += 1

        # --- 軌跡の描画 ---
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], 'c--', linewidth=1.2,
                         alpha=0.5, label='EE Trajectory')

        # --- ロボットアームの描画 ---
        base, j1, j2, ee = self.arm.forward_kinematics()
        points = np.array([base, j1, j2, ee])

        colors = ['#2196F3', '#4CAF50', '#FF9800']
        labels = ['Link1 (Shoulder)', 'Link2 (Elbow)', 'Link3 (Wrist)']
        for i in range(3):
            self.ax.plot([points[i][0], points[i+1][0]],
                         [points[i][1], points[i+1][1]],
                         color=colors[i], linewidth=8, solid_capstyle='round',
                         label=labels[i])

        # 関節マーカー
        joint_labels = ['Base', 'Shoulder', 'Elbow', 'EE']
        joint_colors = ['black', '#2196F3', '#4CAF50', '#FF5722']
        for pt, lbl, col in zip(points, joint_labels, joint_colors):
            self.ax.plot(pt[0], pt[1], 'o', markersize=14, color=col, zorder=5)
            self.ax.annotate(lbl, pt, textcoords="offset points",
                             xytext=(8, 8), fontsize=8, color=col, fontweight='bold')

        # --- グリッパー描画 ---
        gw = self.arm.gripper_width if self.arm.gripper_open else 0.03
        angle = self.arm.angles.theta1 + self.arm.angles.theta2 + self.arm.angles.theta3
        perp = np.array([-np.sin(angle), np.cos(angle)])
        g1 = ee + perp * gw
        g2 = ee - perp * gw
        grip_color = 'gray' if self.arm.gripper_open else 'darkred'
        self.ax.plot([ee[0], g1[0]], [ee[1], g1[1]], color=grip_color, linewidth=5)
        self.ax.plot([ee[0], g2[0]], [ee[1], g2[1]], color=grip_color, linewidth=5)
        gripper_state = "OPEN" if self.arm.gripper_open else "CLOSED"

        # --- ボール描画 ---
        ball_circle = plt.Circle(
            (self.ball.position.x, self.ball.position.y),
            self.ball.radius,
            color=self.ball.color, zorder=4, label='Ball'
        )
        self.ax.add_patch(ball_circle)

        # 配置ターゲットマーカー
        self.ax.plot(self.place_target.x, self.place_target.y,
                     'X', markersize=15, color='purple',
                     label='Place Target', zorder=6)
        target_circle = plt.Circle(
            (self.place_target.x, self.place_target.y),
            self.ball.radius * 1.3,
            color='purple', fill=False, linestyle='--', linewidth=2
        )
        self.ax.add_patch(target_circle)

        # --- ウェイポイント表示 ---
        for i, wp in enumerate(self.waypoints):
            alpha = 0.3 if i < self.current_wp_idx else 0.7
            marker = 'v' if i < self.current_wp_idx else '^'
            self.ax.plot(wp[0], wp[1], marker, markersize=8,
                         color='navy', alpha=alpha)

        # --- 状態テキスト ---
        state_labels = self._state_label_map()
        state_text = state_labels.get(self.state, self.state)
        self.ax.text(0.02, 0.97, f"State: {state_text}",
                     transform=self.ax.transAxes, fontsize=11,
                     verticalalignment='top', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 関節角度テキスト
        angle_text = (
            f"θ1(Shoulder): {np.degrees(self.arm.angles.theta1):+.1f}°\n"
            f"θ2(Elbow):    {np.degrees(self.arm.angles.theta2):+.1f}°\n"
            f"θ3(Wrist):    {np.degrees(self.arm.angles.theta3):+.1f}°\n"
            f"Gripper: {gripper_state}"
        )
        self.ax.text(0.02, 0.78, angle_text,
                     transform=self.ax.transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

        # フレーム情報
        self.ax.text(0.98, 0.97, f"Frame: {frame}",
                     transform=self.ax.transAxes, fontsize=9,
                     horizontalalignment='right', verticalalignment='top',
                     color='gray')

        self.ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        self.ax.set_xlim(-2.5, 3.0)
        self.ax.set_ylim(-0.3, 3.0)
        self.ax.set_title('3-Axis Robot Arm - Ball Grasping Simulation', fontsize=14, fontweight='bold')

        if self.state == SimulationState.DONE:
            self.ax.text(0.5, 0.5, 'GRASP COMPLETE!',
                         transform=self.ax.transAxes,
                         fontsize=28, color='green', fontweight='bold',
                         ha='center', va='center', alpha=0.4)

    # ----------------------------------------------------------
    # シミュレーション実行
    # ----------------------------------------------------------
    def run(self, frames: int = 200, interval: int = 80, save_gif: bool = False):
        """
        アニメーションを起動する。

        Args:
            frames:    総フレーム数
            interval:  フレーム間隔 [ms]
            save_gif:  True なら robot_arm_simulation.gif として保存
        """
        ani = animation.FuncAnimation(
            self.fig, self.update,
            frames=frames, interval=interval,
            repeat=False
        )

        if save_gif:
            self.log("GIFアニメーション保存中...")
            ani.save('robot_arm_simulation.gif', writer='pillow', fps=12)
            self.log("保存完了: robot_arm_simulation.gif")
        else:
            plt.tight_layout()
            plt.show()

# ============================================================
# エントリポイント
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print(" 3軸ロボットアーム ボール把持シミュレーション")
    print(" 3-Axis Robot Arm Ball Grasping Simulation")
    print("=" * 60)
    print()
    print("リンク長設定:")
    print("  L1 (Shoulder-Elbow) = 1.0 m")
    print("  L2 (Elbow-Wrist)    = 0.8 m")
    print("  L3 (Wrist-EE)       = 0.4 m")
    print()
    print("動作フロー:")
    print("  IDLE → MOVING_TO → GRASPING → LIFTING")
    print("       → PLACING → RELEASING → RETURNING → DONE")
    print()

    sim = BallGraspingSimulation()
    # save_gif=True にすると GIF ファイルとして保存されます
    sim.run(frames=200, interval=80, save_gif=False)
