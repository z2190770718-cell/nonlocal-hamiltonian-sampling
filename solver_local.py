# -*- coding: utf-8 -*-
"""Second-order density solver with local KNN and nonlocal modes.

The script entry point runs the separate alpha=2 local baseline; Config retains local_mode=False."""

import numpy as np
import os
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
import time
from sklearn.neighbors import NearestNeighbors
from targets import get_target_distribution


class Config:
    """配置类"""

    def __init__(self, alpha_levy=1.0, target_name='ring_double_well', local_mode=False):
        """
        参数:
        - alpha_levy: Lévy 指数（仅在 nonlocal 模式下有效）
        - target_name: 目标分布名称
        - local_mode: True = 局部图（纯 KNN），False = 非局部图（KNN + Lévy 长边）
        """
        self.Nx, self.Ny = 50, 50
        self.x_min, self.x_max = -5, 5
        self.y_min, self.y_max = -5, 5
        self.dt = 5e-4
        
        self.N = self.Nx * self.Ny
        self.steps = 20000
        self.h = (self.x_max - self.x_min) / self.Nx
        
        # 物理参数
        self.alpha_levy = alpha_levy
        self.eta = 1.0
        self.kappa = 0.05
        
        # ========== 局部模式开关 ==========
        self.local_mode = local_mode
        
        # 阻尼参数
        self.gamma_opt = None  # 由外部设置
        self.gamma_base = 0.1#0.10
        self.gamma_alpha_coeff = 0.08
        
        # 图参数
        beta_local = 0.3
        self.k_neighbors = int(self.N ** beta_local)
        self.vel_max = 8.0
        self.eps = 1e-12
        self.long_jump_threshold = 2.0
        
        # 长边采样参数（仅在 nonlocal 模式下使用）
        self.n_long_per_node = max(1, int(self.N ** (0.3 * (1 - self.alpha_levy / 2))))
        self.n_candidates = self.N - 1
        
        # I/O
        self.save_every = 500
        self.n_samples = 10000
        self.n_tracked = 200
        self.target_name = target_name
        
        # 根据模式设置保存路径
        mode_str = "local" if local_mode else "nonlocal"
        self.save_folder = str(ROOT / "results" / f"{mode_str}_runs" / f"{target_name}_alpha{alpha_levy}")

    def get_alpha_adaptive(self, step):
        """自适应阻尼（适用于 nonlocal 和 local 模式）"""
        gamma = self.gamma_base + self.gamma_alpha_coeff * self.alpha_levy
        
        if step < 8000:
            return 0.20 * gamma / 0.10 + 0.10 * np.exp(-step / 2500.0)
        elif step < 18000:
            return 0.13 * gamma / 0.10
        else:
            return 0.13 * gamma / 0.10 * np.exp(-(step - 18000) / 8000.0)


class Solver:
    def __init__(self, cfg):
        self.cfg = cfg
        self._setup_grid()
        self._setup_target()
        self._setup_initial()
        os.makedirs(self.cfg.save_folder, exist_ok=True)
        self._build_levy_graph()
        self._init_velocity()
        self._init_tracked_particles()

        # ========== 完整历史记录 ==========
        self.history = {
            'step': [], 'KL': [], 'alpha': [], 'u_max': [],
            'kinetic': [], 'action': [],
            'Hamiltonian': [],  # ⭐ 新增
            'transport_length': [],
            'long_flux_ratio': [],
            'mean_jump': [],
            'max_jump': [],
            'flux_norm': [],
            'convection': [],
            'grad_force': [],
            'conv_force': [],
            'entropy_dissipation': [],
        }

        self.snapshot_steps = list(range(0, self.cfg.steps, self.cfg.save_every))
        if self.cfg.steps - 1 not in self.snapshot_steps:
            self.snapshot_steps.append(self.cfg.steps - 1)

        mode_str = "Local" if self.cfg.local_mode else "Nonlocal"
        print(f"Mode: {mode_str}, Grid: {self.cfg.Nx}×{self.cfg.Ny}, Nodes: {self.N}, Edges: {self.E}")

    # ============================================================
    # 初始化
    # ============================================================

    def _setup_grid(self):
        x = np.linspace(self.cfg.x_min, self.cfg.x_max, self.cfg.Nx)
        y = np.linspace(self.cfg.y_min, self.cfg.y_max, self.cfg.Ny)
        self.Xg, self.Yg = np.meshgrid(x, y)
        self.points = np.stack([self.Xg.ravel(), self.Yg.ravel()], axis=1)
        self.N = len(self.points)

    def _setup_target(self):
        target = get_target_distribution(self.cfg.target_name, self.points)
        self.pi = target.get_pi()
        self.target_name = target.get_name()
        self.pi = np.maximum(self.pi, 1e-12)
        self.pi = self.pi / np.sum(self.pi)

    def _setup_initial(self):
        self.rho = np.exp(-0.5 * (self.points[:, 0] ** 2 + self.points[:, 1] ** 2))
        self.rho /= np.sum(self.rho)

    # ============================================================
    # 图构建：Local vs Nonlocal
    # ============================================================

    def _build_levy_graph(self):
        from sklearn.neighbors import NearestNeighbors
        rng = np.random.default_rng(42)

        alpha = self.cfg.alpha_levy
        local_mode = self.cfg.local_mode
        n_long_per_node = self.cfg.n_long_per_node

        # KNN 局部边（两种模式都有）
        nbrs = NearestNeighbors(n_neighbors=self.cfg.k_neighbors, algorithm='kd_tree')
        nbrs.fit(self.points)
        distances, indices = nbrs.kneighbors(self.points)

        edge_pairs = {}
        for i in range(self.N):
            for n in range(1, self.cfg.k_neighbors):
                j = indices[i, n]
                a, b = min(i, j), max(i, j)
                if (a, b) in edge_pairs:
                    continue
                edge_pairs[(a, b)] = [1.0, 1.0, distances[i, n]]

        n_local = sum(1 for v in edge_pairs.values() if v[0] == 1.0)

        # ============================================================
        # 关键分支：只有 nonlocal 模式才添加长边
        # ============================================================
        if not local_mode:
            for i in range(self.N):
                other_nodes = [x for x in range(self.N) if x != i]
                if len(other_nodes) < self.cfg.n_candidates:
                    candidates = other_nodes
                else:
                    candidates = rng.choice(other_nodes, size=self.cfg.n_candidates, replace=False)

                valid_nodes, valid_dists = [], []
                xi = self.points[i]

                for j in candidates:
                    a, b = min(i, j), max(i, j)
                    if (a, b) in edge_pairs:
                        continue
                    dx = xi[0] - self.points[j, 0]
                    dy = xi[1] - self.points[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    valid_nodes.append(j)
                    valid_dists.append(dist)

                if len(valid_nodes) == 0:
                    continue

                d = 2.0
                valid_dists = np.array(valid_dists)
                probs = 1.0 / (valid_dists + 1e-12) ** (d + alpha)
                probs /= np.sum(probs)

                n_sample = min(n_long_per_node, len(valid_nodes))
                sampled_idx = rng.choice(len(valid_nodes), size=n_sample, replace=True, p=probs)

                for idx in sampled_idx:
                    j = valid_nodes[idx]
                    dist = valid_dists[idx]
                    a, b = min(i, j), max(i, j)
                    if (a, b) not in edge_pairs:
                        edge_pairs[(a, b)] = [None, None, dist]
                    if i == a:
                        edge_pairs[(a, b)][0] = probs[idx]
                    else:
                        edge_pairs[(a, b)][1] = probs[idx]

        # ============================================================
        # 构建边数组 + 核对称化（统一处理 nonlocal 和 local）
        # ============================================================
        keys = list(edge_pairs.keys())
        self.edge_i = np.array([k[0] for k in keys], dtype=np.int32)
        self.edge_j = np.array([k[1] for k in keys], dtype=np.int32)
        vals = list(edge_pairs.values())
        self.edge_dist = np.array([v[2] for v in vals])

        # ============================================================
        # 核对称化（与 first_order_solver.py 完全一致）
        # ============================================================
        p_ij = np.array([v[0] if v[0] is not None else 0.0 for v in vals])
        p_ji = np.array([v[1] if v[1] is not None else 0.0 for v in vals])

        local_mask = (p_ij >= 0.999) | (p_ji >= 0.999)

        p_sym = p_ij + p_ji
        p_sym = np.maximum(p_sym, 1e-12)

        importance_weight = 1.0 / (p_sym * n_long_per_node + 1e-12)
        importance_weight[local_mask] = 1.0

        self.E = len(self.edge_i)

        # 预存几何量
        self.edge_dx = self.points[self.edge_j, 0] - self.points[self.edge_i, 0]
        self.edge_dy = self.points[self.edge_j, 1] - self.points[self.edge_i, 1]
        self.edge_unit_x = self.edge_dx / (self.edge_dist + 1e-12)
        self.edge_unit_y = self.edge_dy / (self.edge_dist + 1e-12)

        # ============================================================
        # 计算 nu
        # ============================================================
        d = 2
        base_nu = 1.0 / ((self.edge_dist + 1e-12) ** (d + alpha))

        x_uniq = np.sort(np.unique(self.points[:, 0]))
        y_uniq = np.sort(np.unique(self.points[:, 1]))
        hx = (x_uniq[-1] - x_uniq[0]) / (len(x_uniq) - 1)
        hy = (y_uniq[-1] - y_uniq[0]) / (len(y_uniq) - 1)
        vol = hx * hy

        self.nu = base_nu * importance_weight * vol

        # ============================================================
        # 打印信息
        # ============================================================
        n_long = self.E - n_local
        print(f"\nGraph: {self.N} nodes, {self.E} edges")
        print(f"  Mode: {'LOCAL' if local_mode else 'NONLOCAL'}")
        print(f"  Local: {n_local}, Long: {n_long}")
        print(f"  Max dist: {np.max(self.edge_dist):.2f}")
        print(f"  Importance weight: min={np.min(importance_weight):.3f}, max={np.max(importance_weight):.3f}")
        print(f"  nu: mean={np.mean(self.nu):.4f}, std={np.std(self.nu):.4f}")

        self.nu_global_mean = np.mean(self.nu)
        self.nu_global_std = np.std(self.nu)

        # 保存图结构
        np.savez(f"{self.cfg.save_folder}/graph_structure.npz",
                edge_i=self.edge_i, edge_j=self.edge_j,
                edge_dist=self.edge_dist, nu=self.nu,
                points=self.points, local_mode=local_mode)

    # ============================================================
    # 初始化速度和粒子轨迹
    # ============================================================

    def _init_velocity(self):
        self.u = np.random.randn(self.E) * 1e-6

    def _init_tracked_particles(self):
        self.tracked_indices = np.random.choice(self.N, size=self.cfg.n_tracked, replace=False)
        self.tracked_positions = [self.points[self.tracked_indices].copy()]

    # ============================================================
    # 核心计算函数
    # ============================================================

    def _log_mean(self, a, b):
        a_safe = np.maximum(a, self.cfg.eps)
        b_safe = np.maximum(b, self.cfg.eps)
        
        mask = np.abs(a - b) < 1e-10
        out = (a - b) / (np.log(a_safe) - np.log(b_safe))
        
        if np.any(mask):
            out[mask] = a[mask]
        
        out = np.nan_to_num(out, nan=0.5 * (a + b))
        out = np.where(np.isinf(out), a, out)
        out = np.where(np.isnan(out), 0.5 * (a + b), out)
        
        return out

    def _dtheta_da(self, a, b):
        log_ratio = np.log(a + self.cfg.eps) - np.log(b + self.cfg.eps)
        numerator = log_ratio - (1.0 - b / (a + self.cfg.eps))
        denominator = log_ratio**2 + self.cfg.eps
        out = numerator / denominator
        out[np.abs(a - b) < 1e-10] = 0.5
        return np.nan_to_num(out, nan=0.5)

    def _compute_kinetic(self, theta):
        return 0.5 * np.sum(theta * self.u**2 * self.nu)

    def _compute_node_velocity(self):
        ux = self.u * self.edge_unit_x * self.nu
        uy = self.u * self.edge_unit_y * self.nu
        vx = np.bincount(self.edge_i, weights=ux, minlength=self.N) - \
             np.bincount(self.edge_j, weights=ux, minlength=self.N)
        vy = np.bincount(self.edge_i, weights=uy, minlength=self.N) - \
             np.bincount(self.edge_j, weights=uy, minlength=self.N)
        w = np.bincount(self.edge_i, weights=self.nu, minlength=self.N) + \
            np.bincount(self.edge_j, weights=self.nu, minlength=self.N)
        mask = w > 0
        vx[mask] /= w[mask]
        vy[mask] /= w[mask]
        return vx, vy

    def _sample_particles(self):
        idx = np.random.choice(self.N, size=self.cfg.n_samples, p=self.rho)
        return self.points[idx].copy()

    def _compute_transport_statistics(self, m):
        flux_abs = np.abs(m)
        total_flux = np.sum(flux_abs) + 1e-12

        transport_length = np.sum(flux_abs * self.edge_dist) / total_flux

        long_mask = self.edge_dist > self.cfg.long_jump_threshold
        long_flux_ratio = np.sum(flux_abs[long_mask]) / total_flux

        mean_jump = transport_length
        max_jump = self.edge_dist[np.argmax(flux_abs)]

        return {
            'transport_length': transport_length,
            'long_flux_ratio': long_flux_ratio,
            'mean_jump': mean_jump,
            'max_jump': max_jump,
        }

    # ============================================================
    # 保存函数
    # ============================================================

    def _save_snapshot(self, step, m, theta):
        vx, vy = self._compute_node_velocity()
        long_mask = self.edge_dist > self.cfg.long_jump_threshold
        long_ratio = np.sum(np.abs(m[long_mask])) / (np.sum(np.abs(m)) + 1e-12)

        np.savez(f"{self.cfg.save_folder}/flux_{step:06d}.npz",
                 m=m, u=self.u, theta=theta, vx=vx, vy=vy,
                 edge_i=self.edge_i, edge_j=self.edge_j,
                 edge_dist=self.edge_dist, nu=self.nu,
                 points=self.points, rho=self.rho, pi=self.pi,
                 Xg=self.Xg, Yg=self.Yg,
                 alpha=self.cfg.alpha_levy, step=step,
                 long_jump_ratio=long_ratio)

        rho_grid = self.rho.reshape(self.cfg.Ny, self.cfg.Nx)
        pi_grid = self.pi.reshape(self.cfg.Ny, self.cfg.Nx)
        samples = self._sample_particles()
        np.savez(f"{self.cfg.save_folder}/snapshot_{step:06d}.npz",
                 rho=rho_grid, pi=pi_grid, samples=samples,
                 Xg=self.Xg, Yg=self.Yg, step=step)

        np.savez(f"{self.cfg.save_folder}/trajectory_{step:06d}.npz",
                 positions=self.tracked_positions[-1],
                 indices=self.tracked_indices, step=step)

    def _save_history(self):
        np.savez(f"{self.cfg.save_folder}/history.npz",
                 **{k: np.array(v) for k, v in self.history.items()})

    # ============================================================
    # 主运行函数
    # ============================================================

    def run(self):
        mode_str = "LOCAL" if self.cfg.local_mode else "NONLOCAL"
        print(f"\nRunning {mode_str} mode, α={self.cfg.alpha_levy}, steps={self.cfg.steps}")
        
        if self.cfg.gamma_opt is not None:
            gamma_fixed = self.cfg.gamma_opt
            print(f"  Using fixed damping: gamma = {gamma_fixed:.4f}")
            use_fixed_gamma = True
        else:
            gamma_fixed = None
            print("  Using adaptive damping (underdamped regime)")
            use_fixed_gamma = False
        
        self.pi = np.maximum(self.pi, self.cfg.eps)
        self.pi = self.pi / np.sum(self.pi)

        prev_KL = None

        for step in range(self.cfg.steps):
            t0 = time.time()

            rho_safe = np.maximum(self.rho, self.cfg.eps)
            G = np.log(rho_safe) - np.log(self.pi)

            rho_i = rho_safe[self.edge_i]
            rho_j = rho_safe[self.edge_j]
            G_i = G[self.edge_i]
            G_j = G[self.edge_j]

            theta = self._log_mean(rho_i, rho_j)

            dtheta = self._dtheta_da(rho_i, rho_j)
            edge_contrib = dtheta * (self.u**2) * self.nu
            S = np.bincount(self.edge_i, weights=edge_contrib, minlength=self.N)

            C_raw = 0.5 * (S[self.edge_i] - S[self.edge_j])
            C_scale = np.percentile(np.abs(C_raw), 95) + 1e-12
            c_coeff = 1.0 * np.exp(-step / 6000.0)
            c_coeff = max(c_coeff, 0.4)
            C_edge = c_coeff * np.tanh(C_raw / C_scale)

            if use_fixed_gamma:
                gamma_k = gamma_fixed
            else:
                gamma_k = self.cfg.get_alpha_adaptive(step)

            self.u = (
                self.u
                + self.cfg.dt * ((G_i - G_j) + C_edge)
            ) / (1 + self.cfg.dt * gamma_k)
            self.u = self.cfg.vel_max * np.tanh(self.u / self.cfg.vel_max)

            m = theta * self.u * self.nu
            rho_dot = np.bincount(self.edge_i, weights=-m, minlength=self.N) + \
                    np.bincount(self.edge_j, weights=m, minlength=self.N)
            self.rho += self.cfg.dt * rho_dot
            self.rho = np.maximum(self.rho, self.cfg.eps)
            self.rho /= np.sum(self.rho)

            vx, vy = self._compute_node_velocity()
            new_pos = self.tracked_positions[-1].copy()
            new_pos[:, 0] += self.cfg.dt * vx[self.tracked_indices]
            new_pos[:, 1] += self.cfg.dt * vy[self.tracked_indices]
            new_pos[:, 0] = np.clip(new_pos[:, 0], self.cfg.x_min, self.cfg.x_max)
            new_pos[:, 1] = np.clip(new_pos[:, 1], self.cfg.y_min, self.cfg.y_max)
            self.tracked_positions.append(new_pos)

            KL = np.sum(self.rho * (np.log(self.rho + self.cfg.eps) - np.log(self.pi)))
            kinetic = self._compute_kinetic(theta)
            Hamiltonian = KL + kinetic  # ⭐ 总能量

            flux_norm = np.sum(np.abs(m))
            grad_force = np.mean(np.abs(G_i - G_j))
            conv_force = np.mean(np.abs(C_edge))
            dKL = (prev_KL - KL) / self.cfg.dt if prev_KL is not None else 0.0

            trans_stats = self._compute_transport_statistics(m)

            self.history['step'].append(step)
            self.history['KL'].append(KL)
            self.history['u_max'].append(np.max(np.abs(self.u)))
            self.history['kinetic'].append(kinetic)
            self.history['Hamiltonian'].append(Hamiltonian)  # ⭐ 新增
            self.history['action'].append(np.sum(theta * self.u**2 * self.nu))
            self.history['transport_length'].append(trans_stats['transport_length'])
            self.history['long_flux_ratio'].append(trans_stats['long_flux_ratio'])
            self.history['mean_jump'].append(trans_stats['mean_jump'])
            self.history['max_jump'].append(trans_stats['max_jump'])
            self.history['flux_norm'].append(flux_norm)
            self.history['convection'].append(conv_force)
            self.history['grad_force'].append(grad_force)
            self.history['conv_force'].append(conv_force)
            self.history['entropy_dissipation'].append(dKL)

            prev_KL = KL

            if step in self.snapshot_steps:
                self._save_snapshot(step, m, theta)

            if step % 500 == 0 or step == self.cfg.steps - 1:
                print(
                    f"  step {step:6d} | KL={KL:.2e} | "
                    f"L={trans_stats['transport_length']:.3f} | "
                    f"long_ratio={trans_stats['long_flux_ratio']:.4f} | "
                    f"grad={grad_force:.3e} | "
                    f"conv={conv_force:.3e} | "
                    f"time={1000*(time.time()-t0):.1f}ms"
                )

        self._save_history()
        np.savez(f"{self.cfg.save_folder}/tracked_trajectories.npz",
                 positions=np.array(self.tracked_positions),
                 indices=self.tracked_indices)

        print(f"\n  Saved to: {self.cfg.save_folder}")
        return self.history


# ============================================================
# 主程序
# ============================================================


def main():
    """Run the archived experiment defaults, with optional explicit overrides."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--target", default="ring_double_well")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional NumPy seed; unset preserves archived behavior. Graph seed remains 42.")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    cfg = Config(alpha_levy=args.alpha, target_name=args.target, local_mode=True)
    cfg.gamma_opt = None
    cfg.steps = args.steps
    if args.output_dir is not None:
        cfg.save_folder = str(args.output_dir)
    solver = Solver(cfg)
    solver.run()


if __name__ == "__main__":
    main()
