# 3軸ロボットアーム ボール把持シミュレーション
# 3-Axis Robot Arm Ball Grasping Simulation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 概要 / Overview

Python + matplotlib を用いた **3軸ロボットアーム** によるボール把持シミュレーションです。  
逆運動学（ヤコビアン擬似逆行列法）でリアルタイムに関節角度を計算し、  
アームがボールを掴んで目標位置へ運ぶ一連の動作をアニメーションで可視化します。

---

## ファイル構成

```
3axis-robot-arm-simulation/
├── robot_arm_simulation.py   # メインシミュレーション（アニメーション付き）
├── kinematics.py             # 順・逆運動学ユーティリティ（独立モジュール）
├── requirements.txt          # 依存ライブラリ
└── README.md
```

---

## 動作要件 / Requirements

- Python 3.9+
- numpy
- matplotlib
- pillow（GIF保存時のみ）

```bash
pip install -r requirements.txt
```

---

## 実行方法 / Usage

### シミュレーション実行（アニメーション表示）

```bash
python robot_arm_simulation.py
```

### GIFアニメーションとして保存

`robot_arm_simulation.py` の最終行を以下に変更:

```python
sim.run(frames=200, interval=80, save_gif=True)
```

### 運動学テスト

```bash
python kinematics.py
```

---

## アーム仕様 / Robot Arm Specs

| パラメータ | 値 |
|---|---|
| 自由度 | 3 DOF (2D平面) |
| L1 (肩-肘) | 1.0 m |
| L2 (肘-手首) | 0.8 m |
| L3 (手首-EE) | 0.4 m |
| 最大リーチ | 2.2 m |
| 逆運動学手法 | Jacobian Pseudo-Inverse |
| 関節数 | θ1 (Shoulder) / θ2 (Elbow) / θ3 (Wrist) |

---

## 動作フロー / State Machine

```
IDLE
  └─► MOVING_TO  (ボール上空アプローチ)
        └─► GRASPING   (把持位置まで降下 → グリッパー閉)
              └─► LIFTING    (リフトアップ)
                    └─► PLACING    (配置先上空へ移動)
                          └─► RELEASING  (降下 → グリッパー開)
                                └─► RETURNING  (初期姿勢へ)
                                      └─► DONE
```

---

## 逆運動学アルゴリズム / IK Algorithm

**ヤコビアン擬似逆行列法 (Jacobian Pseudo-Inverse)**

```
Δθ = α · J⁺ · e
```

- `e` : エンドエフェクタ位置誤差 (target - current)
- `J` : ヤコビアン行列 (2×3)
- `J⁺` : Mooreペンローズ擬似逆行列
- `α` : ステップサイズ（学習率）

解析的ヤコビアンは `kinematics.py` の `jacobian_3dof()` に実装済み。

---

## スクリーンショット

```
[アニメーション実行時の画面]
- 青リンク: 上腕 (Shoulder-Elbow)
- 緑リンク: 前腕 (Elbow-Wrist)  
- 橙リンク: 手首 (Wrist-EE)
- オレンジ円: ボール（把持後: 赤→配置後: 緑）
- 紫 ×マーク: 配置目標位置
- シアン破線: エンドエフェクタ軌跡
```

---

## ライセンス / License

MIT License

---

*Created by SSG-0123 | Kumamoto National College of Technology*
