#!/usr/bin/env python3
"""
kinematics.py - 3軸ロボットアーム 運動学ユーティリティ

順運動学 (FK) および 逆運動学 (IK) の独立モジュール。
ロボットアームのパラメータを変更してテスト可能。
"""

import numpy as np
from typing import Tuple


def forward_kinematics_2d(
    theta1: float, theta2: float, theta3: float,
    L1: float = 1.0, L2: float = 0.8, L3: float = 0.4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    2D平面上の順運動学。

    Args:
        theta1, theta2, theta3: 各関節角度 [rad]
        L1, L2, L3: 各リンク長 [m]

    Returns:
        base, joint1, joint2, end_effector の座標 (numpy array [x, y])
    """
    base = np.array([0.0, 0.0])
    j1 = base + np.array([L1 * np.cos(theta1),
                            L1 * np.sin(theta1)])
    j2 = j1 + np.array([L2 * np.cos(theta1 + theta2),
                          L2 * np.sin(theta1 + theta2)])
    ee = j2 + np.array([L3 * np.cos(theta1 + theta2 + theta3),
                          L3 * np.sin(theta1 + theta2 + theta3)])
    return base, j1, j2, ee


def jacobian_3dof(
    theta1: float, theta2: float, theta3: float,
    L1: float = 1.0, L2: float = 0.8, L3: float = 0.4
) -> np.ndarray:
    """
    3自由度2Dアームのヤコビアン行列 (2x3) を解析的に計算。

    J = d(EE_pos) / d(theta)
    """
    t12  = theta1 + theta2
    t123 = theta1 + theta2 + theta3

    J = np.array([
        [
            -L1 * np.sin(theta1) - L2 * np.sin(t12) - L3 * np.sin(t123),
            -L2 * np.sin(t12)    - L3 * np.sin(t123),
            -L3 * np.sin(t123)
        ],
        [
            L1 * np.cos(theta1) + L2 * np.cos(t12) + L3 * np.cos(t123),
            L2 * np.cos(t12)    + L3 * np.cos(t123),
            L3 * np.cos(t123)
        ]
    ])
    return J


def ik_jacobian(
    target_x: float, target_y: float,
    L1: float = 1.0, L2: float = 0.8, L3: float = 0.4,
    init_angles: Tuple[float, float, float] = (np.pi/4, -np.pi/4, 0.0),
    max_iter: int = 1000, tol: float = 1e-5, alpha: float = 0.3
) -> Tuple[bool, float, float, float]:
    """
    ヤコビアン擬似逆行列法による逆運動学ソルバー。

    Args:
        target_x, target_y: 目標エンドエフェクタ座標
        init_angles: 初期関節角度 (theta1, theta2, theta3)
        max_iter: 最大反復回数
        tol: 収束判定閾値
        alpha: ステップサイズ（学習率）

    Returns:
        (converged, theta1, theta2, theta3)
    """
    t1, t2, t3 = init_angles
    target = np.array([target_x, target_y])

    for i in range(max_iter):
        _, _, _, ee = forward_kinematics_2d(t1, t2, t3, L1, L2, L3)
        error = target - ee

        if np.linalg.norm(error) < tol:
            return True, t1, t2, t3

        J = jacobian_3dof(t1, t2, t3, L1, L2, L3)
        J_pinv = np.linalg.pinv(J)
        dtheta = alpha * J_pinv @ error

        t1 += dtheta[0]
        t2 += dtheta[1]
        t3 += dtheta[2]

    return False, t1, t2, t3


def workspace_reachable(
    x: float, y: float,
    L1: float = 1.0, L2: float = 0.8, L3: float = 0.4
) -> bool:
    """指定座標がアームの到達範囲内かチェック"""
    max_r = L1 + L2 + L3
    min_r = abs(L1 - L2 - L3)
    dist  = np.hypot(x, y)
    return min_r <= dist <= max_r


if __name__ == '__main__':
    print("=== Forward Kinematics Test ===")
    angles = (np.pi/4, -np.pi/3, np.pi/6)
    base, j1, j2, ee = forward_kinematics_2d(*angles)
    print(f"  Base      : {base}")
    print(f"  Joint1    : {j1}")
    print(f"  Joint2    : {j2}")
    print(f"  End Eff.  : {ee}")

    print("\n=== Jacobian Test ===")
    J = jacobian_3dof(*angles)
    print(f"  J =\n{J}")

    print("\n=== Inverse Kinematics Test ===")
    target = (1.5, 0.8)
    converged, t1, t2, t3 = ik_jacobian(*target)
    _, _, _, ee_ik = forward_kinematics_2d(t1, t2, t3)
    print(f"  Target: {target}")
    print(f"  Converged: {converged}")
    print(f"  Angles: θ1={np.degrees(t1):.2f}°, θ2={np.degrees(t2):.2f}°, θ3={np.degrees(t3):.2f}°")
    print(f"  Achieved EE: ({ee_ik[0]:.4f}, {ee_ik[1]:.4f})")
    print(f"  Error: {np.linalg.norm(np.array(target) - ee_ik):.6f} m")
