#!/usr/bin/env python3
"""
kinematics_3d.py
================
3D空間 6DOFロボットアーム 運動学ユーティリティ

機能一覧
--------
- DH 変換行列の生成
- 順運動学（FK）: 廂数積 DH 変換行列
- 解析的ヤコビアン (3x6) 計算
- Damped Least Squares IKソルバー
- 到達可能内益チェック
- ユニットテスト
"""

import numpy as np
from typing import List, Tuple

DEG = np.pi / 180


# ============================================================
# DH 変換行列
# ============================================================

def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    標準 Denavit–Hartenberg 変換行列 (4x4) を返す。

    | cosθ  -sinθ·cosα   sinθ·sinα   a·cosθ |
    | sinθ   cosθ·cosα  -cosθ·sinα   a·sinθ |
    |   0      sinα         cosα         d    |
    |   0        0             0           1    |
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,  -st*ca,  st*sa,  a*ct],
        [st,   ct*ca, -ct*sa,  a*st],
        [0,      sa,    ca,     d  ],
        [0,       0,     0,     1  ],
    ], dtype=float)


# ============================================================
# 6DOF FK
# ============================================================

def fk_6dof(
    q: np.ndarray,
    D1: float = 0.20,
    A2: float = 0.50,
    A3: float = 0.40,
    D4: float = 0.10,
    D6: float = 0.12,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    6DOFアームの順運動学。

    DHテーブル (theta, d, a, alpha):
      J1: (q[0],  D1,  0,   pi/2)
      J2: (q[1],   0,  A2,  0   )
      J3: (q[2],   0,  A3,  0   )
      J4: (q[3],  D4,  0,   pi/2)
      J5: (q[4],   0,  0,  -pi/2)
      J6: (q[5],  D6,  0,   0   )

    Returns:
        positions: 各関節座標リスト (7点, 各要素 shape=(3,))
        T_ee:      エンドエフェクタ変換行列 (4x4)
    """
    pi2 = np.pi / 2
    params = [
        (q[0],  D1,  0,    pi2),
        (q[1],   0,  A2,   0  ),
        (q[2],   0,  A3,   0  ),
        (q[3],  D4,  0,    pi2),
        (q[4],   0,  0,   -pi2),
        (q[5],  D6,  0,    0  ),
    ]
    T = np.eye(4)
    positions = [T[:3, 3].copy()]
    for p in params:
        T = T @ dh_transform(*p)
        positions.append(T[:3, 3].copy())
    return positions, T


# ============================================================
# 3Dエンドエフェクタヤコビアン (3x6)
# ============================================================

def jacobian_3d_6dof(
    q: np.ndarray,
    D1: float = 0.20,
    A2: float = 0.50,
    A3: float = 0.40,
    D4: float = 0.10,
    D6: float = 0.12,
    delta: float = 1e-6,
) -> np.ndarray:
    """
    6DOFアームのエンドエフェクタ位置ヤコビアン (3x6)。
    数値微分で計算。
    """
    _, T0 = fk_6dof(q, D1, A2, A3, D4, D6)
    ee0   = T0[:3, 3]
    J     = np.zeros((3, 6))
    for i in range(6):
        dq     = q.copy()
        dq[i] += delta
        _, T1  = fk_6dof(dq, D1, A2, A3, D4, D6)
        J[:, i] = (T1[:3, 3] - ee0) / delta
    return J


# ============================================================
# Damped Least Squares IK
# ============================================================

def ik_dls(
    target: np.ndarray,
    q_init: Optional_ndarray = None,
    D1: float = 0.20,
    A2: float = 0.50,
    A3: float = 0.40,
    D4: float = 0.10,
    D6: float = 0.12,
    max_iter: int = 1000,
    tol:      float = 1e-5,
    alpha:    float = 0.8,
    lam:      float = 0.01,
) -> Tuple[bool, np.ndarray]:
    """
    Damped Least Squares IKソルバー。

    更新式:
        Δq = α J^T (J J^T + λ^2 I)^{-1} e

    Args:
        target  : 目標 EE 位置 [x, y, z]
        q_init  : 初期関節角度 (None の場合はゼロ始まり)
        max_iter: 最大反復回数
        tol     : 収束判定閾値 [m]
        alpha   : ステップサイズ
        lam     : ダンピング係数

    Returns:
        (converged, q_solution)
    """
    if q_init is None:
        q = np.zeros(6)
    else:
        q = q_init.copy()

    for i in range(max_iter):
        _, T = fk_6dof(q, D1, A2, A3, D4, D6)
        err  = target - T[:3, 3]
        if np.linalg.norm(err) < tol:
            return True, q

        J    = jacobian_3d_6dof(q, D1, A2, A3, D4, D6)
        JJT  = J @ J.T
        dq   = alpha * J.T @ np.linalg.solve(JJT + lam**2 * np.eye(3), err)
        q   += dq

    return False, q


# ============================================================
# 到達可能内益
# ============================================================

def workspace_check(
    pos: np.ndarray,
    A2: float = 0.50,
    A3: float = 0.40,
    D4: float = 0.10,
    D6: float = 0.12,
) -> bool:
    """指定3D座標が到達可能範囲内かチェック"""
    max_r = A2 + A3 + D4 + D6
    return float(np.linalg.norm(pos)) <= max_r * 0.97


# ============================================================
# ユニットテスト
# ============================================================

if __name__ == '__main__':
    # Optional型ヒント代わり
    from typing import Optional as Optional_ndarray
    print("=" * 55)
    print(" kinematics_3d.py ユニットテスト")
    print("=" * 55)

    # --- FK テスト ---
    print("\n[FK Test]")
    q_test = np.array([0, 45, -60, 0, 30, 0]) * DEG
    pos_list, T_ee = fk_6dof(q_test)
    for i, p in enumerate(pos_list):
        name = ['Base','J1','J2','J3','J4','J5','EE'][i]
        print(f"  {name:4s}: ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
    print(f"  T_ee[:3,3] = {T_ee[:3,3]}")

    # --- Jacobian テスト ---
    print("\n[Jacobian Test]")
    J = jacobian_3d_6dof(q_test)
    print(f"  J (3x6) =\n{np.round(J, 4)}")

    # --- IK テスト ---
    print("\n[IK Test]")
    targets = [
        np.array([ 0.8,  0.3,  0.4]),
        np.array([-0.5,  0.6,  0.3]),
        np.array([ 0.6, -0.4,  0.2]),
    ]
    for tgt in targets:
        conv, q_sol = ik_dls(tgt)
        _, T_sol = fk_6dof(q_sol)
        ee_sol   = T_sol[:3, 3]
        err      = np.linalg.norm(tgt - ee_sol)
        reachable = workspace_check(tgt)
        print(f"  target={np.round(tgt,3)}  conv={conv}  "
              f"err={err:.6f}m  reachable={reachable}")
        print(f"    q[deg]={np.round(np.degrees(q_sol),1)}")

    # --- 到達可能範囲チェック ---
    print("\n[到達可能範囲 Test]")
    test_pts = [
        np.array([0.5, 0.5, 0.3]),
        np.array([2.0, 0.0, 0.0]),  # 範囲外
    ]
    for pt in test_pts:
        print(f"  {np.round(pt,2)} -> reachable: {workspace_check(pt)}")
