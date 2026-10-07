# targets.py
"""Synthetic target densities and target factory retained from the research archive."""

import numpy as np
import os
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

class TargetDistribution:
    """目标分布基类"""

    def __init__(self, points):
        self.points = points
        self.pi = None
        self.name = "Base Distribution"

    def compute(self):
        raise NotImplementedError

    def get_pi(self):
        if self.pi is None:
            self.pi = self.compute()
            self.pi += 1e-12
            self.pi /= np.sum(self.pi)
        return self.pi

    def get_name(self):
        return self.name

    def get_density_func(self):
        """返回函数形式供绘图使用"""
        return None


# ============================================================
# 1. 环状双峰（原始）
# ============================================================

class RingDoubleWell(TargetDistribution):
    """环状双峰分布"""

    def __init__(self, points):
        super().__init__(points)
        self.name = "Ring + Double Well"

    def target_density(self, x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        ring = np.exp(-2.0 * (r - 3.0) ** 2)
        well1 = np.exp(-2.0 * (x - 3.0) ** 2)
        well2 = np.exp(-2.0 * (x + 3.0) ** 2)
        return ring * (well1 + well2)

    def compute(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        return self.target_density(x, y)

    def get_density_func(self):
        return self.target_density


# ============================================================
# 2. 双香蕉分布（你要的）
# ============================================================

class DoubleBanana(TargetDistribution):
    """双香蕉形分布"""

    def __init__(self, points):
        super().__init__(points)
        self.name = "Double Banana"

    def target_density(self, x, y):
        # 香蕉形：y ≈ x^2 附近的高斯分布
        banana_left = np.exp(-0.5 * ((y - 0.5 * (x + 3) ** 2) ** 2) / 0.5 ** 2) * \
                      np.exp(-0.5 * ((x + 3) ** 2) / 2 ** 2)
        banana_right = np.exp(-0.5 * ((y - 0.5 * (x - 3) ** 2) ** 2) / 0.5 ** 2) * \
                       np.exp(-0.5 * ((x - 3) ** 2) / 2 ** 2)
        return banana_left + banana_right + 0.01 * np.exp(-0.5 * (x ** 2 + y ** 2) / 3 ** 2)

    def compute(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        return self.target_density(x, y)

    def get_density_func(self):
        return self.target_density


# ============================================================
# 3. 单峰高斯
# ============================================================

class GaussianSinglePeak(TargetDistribution):
    """简单单峰高斯分布"""

    def __init__(self, points, mean=(0, 0), sigma=1.5):
        super().__init__(points)
        self.mean = mean
        self.sigma = sigma
        self.name = f"Gaussian (μ={mean}, σ={sigma})"

    def target_density(self, x, y):
        return np.exp(-0.5 * ((x - self.mean[0]) ** 2 + (y - self.mean[1]) ** 2) / self.sigma ** 2)

    def compute(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        return self.target_density(x, y)

    def get_density_func(self):
        return self.target_density


# ============================================================
# 4. 双峰高斯（无环）
# ============================================================

class DoubleGaussian(TargetDistribution):
    """两个分离的高斯分布"""

    def __init__(self, points, center1=(-3, 0), center2=(3, 0), sigma=1.0):
        super().__init__(points)
        self.center1 = center1
        self.center2 = center2
        self.sigma = sigma
        self.name = f"Double Gaussian (centers={center1[0]}, {center2[0]})"

    def target_density(self, x, y):
        g1 = np.exp(-0.5 * ((x - self.center1[0]) ** 2 + (y - self.center1[1]) ** 2) / self.sigma ** 2)
        g2 = np.exp(-0.5 * ((x - self.center2[0]) ** 2 + (y - self.center2[1]) ** 2) / self.sigma ** 2)
        return g1 + g2

    def compute(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        return self.target_density(x, y)

    def get_density_func(self):
        return self.target_density


# ============================================================
# 5. 各向异性高斯
# ============================================================

class AnisotropicGaussian(TargetDistribution):
    """各向异性高斯分布"""

    def __init__(self, points, mean=(1, 1), cov=((10, 0), (0, 0.05))):
        super().__init__(points)
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.inv_cov = np.linalg.inv(cov)
        self.name = f"Anisotropic Gaussian (cov={cov[0][0]}, {cov[1][1]})"

    def target_density(self, x, y):
        diff_x = x - self.mean[0]
        diff_y = y - self.mean[1]
        exponent = -0.5 * (self.inv_cov[0, 0] * diff_x ** 2 +
                           2 * self.inv_cov[0, 1] * diff_x * diff_y +
                           self.inv_cov[1, 1] * diff_y ** 2)
        return np.exp(exponent)

    def compute(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        return self.target_density(x, y)

    def get_density_func(self):
        return self.target_density


# ============================================================
# 6. 单环分布
# ============================================================

class SingleRing(TargetDistribution):
    """简单环状分布（无双峰）"""

    def __init__(self, points, radius=3.0, width=1.0):
        super().__init__(points)
        self.radius = radius
        self.width = width
        self.name = f"Single Ring (r={radius})"

    def target_density(self, x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        return np.exp(-2.0 * (r - self.radius) ** 2 / self.width ** 2)

    def compute(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        return self.target_density(x, y)

    def get_density_func(self):
        return self.target_density


# ============================================================
# ============================================================
# 7. 四峰高斯混合分布（论文 Figure 6）
# ============================================================

class FourPeaks(TargetDistribution):
    """
    四峰高斯混合分布（论文 Figure 6）
    
    p(x) = 1/4 * Σ_{i=1}^4 N(x; α_i, Σ_i)
    α_i = 2(sin(2(i-1)π/4), cos(2(i-1)π/4))
    Σ_i = 0.25 * I_2
    
    四个中心: (0, 2), (2, 0), (0, -2), (-2, 0)
    标准差: σ = 0.5
    """

    def __init__(self, points):
        super().__init__(points)
        self.name = "Four Gaussian Mixture"
        self.sigma = 0.5
        self.weight = 1.0 / 4.0
        self.centers = self._compute_centers()

    def _compute_centers(self):
        """计算四个中心点：半径为2的圆上均匀分布"""
        centers = []
        for i in range(1, 5):
            angle = 2 * (i - 1) * np.pi / 4
            x = 2 * np.sin(angle)
            y = 2 * np.cos(angle)
            centers.append((x, y))
        return centers

    def target_density(self, x, y):
        rho = np.zeros_like(x)
        for cx, cy in self.centers:
            dist2 = (x - cx) ** 2 + (y - cy) ** 2
            rho += self.weight * np.exp(-dist2 / (2 * self.sigma ** 2))
        return rho

    def compute(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        return self.target_density(x, y)

    def get_density_func(self):
        return self.target_density

    def get_centers(self):
        """返回四个中心点"""
        return self.centers
# ============================================================


# ============================================================
# 8. 高斯混合分布 (灵活的多模态)
# ============================================================

class GaussianMixture(TargetDistribution):
    """
    灵活的高斯混合分布
    
    支持:
    - 任意数量的高斯分量
    - 不同权重
    - 各向同性或各向异性协方差
    
    示例:
        # 8峰规则排列
        centers = [(-4,-4), (-4,0), (-4,4), (0,-4), (0,4), (4,-4), (4,0), (4,4)]
        weights = [1/8] * 8
        sigma = 0.6
        
        # 非对称7峰
        centers = [(-5,-3), (-4,4), (-1,0), (2,-2), (5,3), (1,5), (-5,1)]
        weights = [0.18, 0.08, 0.22, 0.15, 0.17, 0.10, 0.10]
        sigma = 0.5
    """
    
    def __init__(self, points, centers=None, weights=None, sigma=0.6, cov=None):
        """
        参数:
            points: 网格点 (N, 2)
            centers: 高斯中心列表 [(x1,y1), (x2,y2), ...]
            weights: 每个分量的权重 (归一化自动处理)
            sigma: 各向同性标准差 (如果 cov=None)
            cov: 每个分量的协方差矩阵列表，或统一的协方差矩阵
        """
        super().__init__(points)
        
        # 默认使用 8 峰规则排列
        if centers is None:
            centers = [
                (-4, -4), (-4, 0), (-4, 4),
                (0, -4),           (0, 4),
                (4, -4), (4, 0), (4, 4)
            ]
        
        self.centers = np.array(centers)
        self.n_components = len(self.centers)
        
        # 权重处理
        if weights is None:
            self.weights = np.ones(self.n_components) / self.n_components
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / np.sum(self.weights)  # 归一化
        
        # 协方差处理
        self.cov = cov
        self.sigma = sigma
        
        # 构建名称
        self.name = f"Gaussian Mixture ({self.n_components} components)"
        
        # 计算目标分布
        self._compute_distribution()
    
    def _compute_distribution(self):
        """计算混合分布"""
        x = self.points[:, 0]
        y = self.points[:, 1]
        
        rho = np.zeros(len(x))
        
        for i, (cx, cy) in enumerate(self.centers):
            if self.cov is None:
                # 各向同性高斯
                dist2 = (x - cx)**2 + (y - cy)**2
                gaussian = np.exp(-dist2 / (2 * self.sigma**2))
            else:
                # 各向异性高斯 (支持不同协方差)
                if isinstance(self.cov, list):
                    cov_i = self.cov[i] if i < len(self.cov) else self.cov[0]
                else:
                    cov_i = self.cov
                
                # 2x2 协方差矩阵
                inv_cov = np.linalg.inv(cov_i)
                dx = x - cx
                dy = y - cy
                exponent = -0.5 * (inv_cov[0,0] * dx**2 + 
                                   2 * inv_cov[0,1] * dx * dy + 
                                   inv_cov[1,1] * dy**2)
                gaussian = np.exp(exponent)
            
            rho += self.weights[i] * gaussian
        
        self.rho_flat = rho
        self.rho = rho.reshape(self.points.shape[0], 1) if len(self.points.shape) > 1 else rho
    
    def compute(self):
        """返回未归一化的密度"""
        return self.rho_flat
    
    def get_density_func(self):
        """返回密度函数供绘图使用"""
        def density(x, y):
            result = np.zeros_like(x)
            for i, (cx, cy) in enumerate(self.centers):
                if self.cov is None:
                    dist2 = (x - cx)**2 + (y - cy)**2
                    result += self.weights[i] * np.exp(-dist2 / (2 * self.sigma**2))
                else:
                    cov_i = self.cov[i] if isinstance(self.cov, list) and i < len(self.cov) else self.cov
                    inv_cov = np.linalg.inv(cov_i)
                    dx = x - cx
                    dy = y - cy
                    exponent = -0.5 * (inv_cov[0,0] * dx**2 + 
                                       2 * inv_cov[0,1] * dx * dy + 
                                       inv_cov[1,1] * dy**2)
                    result += self.weights[i] * np.exp(exponent)
            return result
        return density
    
    def get_centers(self):
        """返回高斯中心"""
        return self.centers
    
    def get_weights(self):
        """返回权重"""
        return self.weights


# ============================================================
# 9. 非对称高斯混合 (论文推荐)
# ============================================================

class AsymmetricGaussianMixture(GaussianMixture):
    """
    非对称高斯混合分布
    更接近真实数据的复杂多模态分布
    
    7个非对称排列的高斯分量，不同权重
    """
    
    def __init__(self, points):
        # 非对称中心 (避免规则排列)
        centers = [
            (-5, -3),   # 左下
            (-4, 4),    # 左上
            (-1, 0),    # 中心偏左
            (2, -2),    # 右下
            (5, 3),     # 右上
            (1, 5),     # 上
            (-5, 1),    # 左
        ]
        
        # 不同权重 (总和为1)
        weights = [0.18, 0.08, 0.22, 0.15, 0.17, 0.10, 0.10]
        
        # 不同标准差 (各向异性)
        sigmas = [0.5, 0.8, 0.4, 0.6, 0.5, 0.9, 0.7]
        
        # 使用各向同性但不同sigma (简化版)
        # 对于更复杂的各向异性，可以用 cov 参数
        super().__init__(points, centers=centers, weights=weights, sigma=0.5)
        self.name = "Asymmetric Gaussian Mixture (7 components)"
        
        # 重新计算分布 (使用不同的sigma)
        x = self.points[:, 0]
        y = self.points[:, 1]
        rho = np.zeros(len(x))
        
        for i, (cx, cy) in enumerate(self.centers):
            sigma_i = sigmas[i] if i < len(sigmas) else 0.5
            dist2 = (x - cx)**2 + (y - cy)**2
            rho += self.weights[i] * np.exp(-dist2 / (2 * sigma_i**2))
        
        self.rho_flat = rho
        self.rho = rho.reshape(self.points.shape[0], 1) if len(self.points.shape) > 1 else rho
    # ============================================================
# 9. Müller-Brown 势能（标准采样 benchmark）
# ============================================================

class MullerBrownTarget(TargetDistribution):
    """
    Müller-Brown 势能对应的 Boltzmann 分布
    
    标准二维 benchmark 势能，用于验证增强采样方法。
    有三个最小值: A(-0.558, 1.442), B(-0.050, 0.467), C(0.623, 0.028)
    
    参考文献:
        [41] Müller and Brown, 1979
        [42] 增强采样方法的标准测试势能
    """
    
    def __init__(self, points, beta=0.02):
        """
        参数:
            points: 网格点 (N, 2)，x, y 坐标
            beta: 逆温度 1/kT (默认 1.0)
        """
        super().__init__(points)
        self.beta = beta
        self.name = "Müller-Brown Potential"
        
        # 存储网格信息
        self.x_min, self.x_max = -1.5, 1.0
        self.y_min, self.y_max = -0.5, 2.0
        
        # 计算势能和分布
        self._compute_potential()
        self._compute_pi()
    
    def _compute_potential(self):
        """计算 Müller-Brown 势能 U(x,y)"""
        x = self.points[:, 0]
        y = self.points[:, 1]
        
        # 参数 (来自文献公式17)
        a = np.array([-1.0, -1.0, -6.5, 0.7])
        b = np.array([0.0, 0.0, 11.0, 0.6])
        c = np.array([-10.0, -10.0, -6.5, 0.7])
        d = np.array([-200.0, -100.0, -170.0, 15.0])
        x0 = np.array([1.0, 0.0, -0.5, -1.0])
        y0 = np.array([0.0, 0.5, 1.5, 1.0])
        
        # 计算势能: U = Σ d_j * exp(a_j*(x-x0_j)^2 + b_j*(x-x0_j)*(y-y0_j) + c_j*(y-y0_j)^2)
        U = np.zeros_like(x)
        for j in range(4):
            dx = x - x0[j]
            dy = y - y0[j]
            exponent = a[j] * dx**2 + b[j] * dx * dy + c[j] * dy**2
            U += d[j] * np.exp(exponent)
        
        self.U_flat = U
        self.U_min = np.min(U)
        self.U_max = np.max(U)
    
    def _compute_pi(self):
        """计算 Boltzmann 分布 pi ∝ exp(-βU)"""
        log_pi = -self.beta * self.U_flat
        log_pi = log_pi - np.max(log_pi)  # 数值稳定
        self.pi = np.exp(log_pi)
        self.pi /= np.sum(self.pi)
    
    def compute(self):
        """返回未归一化的密度"""
        return np.exp(-self.beta * self.U_flat)
    
    def get_potential(self):
        """返回势能值 (扁平)"""
        return self.U_flat
    
    def get_potential_grid(self, grid_size=None):
        """返回网格上的势能"""
        if grid_size is None:
            # 从 points 推断网格
            x_uniq = np.sort(np.unique(self.points[:, 0]))
            y_uniq = np.sort(np.unique(self.points[:, 1]))
        else:
            x_uniq = np.linspace(self.x_min, self.x_max, grid_size)
            y_uniq = np.linspace(self.y_min, self.y_max, grid_size)
        
        X, Y = np.meshgrid(x_uniq, y_uniq, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel()])
        
        # 计算势能
        a = np.array([-1.0, -1.0, -6.5, 0.7])
        b = np.array([0.0, 0.0, 11.0, 0.6])
        c = np.array([-10.0, -10.0, -6.5, 0.7])
        d = np.array([-200.0, -100.0, -170.0, 15.0])
        x0 = np.array([1.0, 0.0, -0.5, -1.0])
        y0 = np.array([0.0, 0.5, 1.5, 1.0])
        
        U = np.zeros(len(points))
        for j in range(4):
            dx = points[:, 0] - x0[j]
            dy = points[:, 1] - y0[j]
            exponent = a[j] * dx**2 + b[j] * dx * dy + c[j] * dy**2
            U += d[j] * np.exp(exponent)
        
        return X, Y, U.reshape(X.shape)
    
    def get_minima(self):
        """返回三个最小值的位置"""
        return [
            ('A', -0.558, 1.442),
            ('B', -0.050, 0.467),
            ('C', 0.623, 0.028)
        ]
    
    def get_density_func(self):
        """返回密度函数供绘图使用"""
        def density(x, y):
            # 计算势能
            a = np.array([-1.0, -1.0, -6.5, 0.7])
            b = np.array([0.0, 0.0, 11.0, 0.6])
            c = np.array([-10.0, -10.0, -6.5, 0.7])
            d = np.array([-200.0, -100.0, -170.0, 15.0])
            x0 = np.array([1.0, 0.0, -0.5, -1.0])
            y0 = np.array([0.0, 0.5, 1.5, 1.0])
            
            U = np.zeros_like(x)
            for j in range(4):
                dx = x - x0[j]
                dy = y - y0[j]
                exponent = a[j] * dx**2 + b[j] * dx * dy + c[j] * dy**2
                U += d[j] * np.exp(exponent)
            
            return np.exp(-self.beta * U)
        return density
    
    def get_fes(self, grid_size=65):
        """返回自由能面 (用于可视化)"""
        X, Y, U = self.get_potential_grid(grid_size)
        FES = U - np.min(U)  # 归一化
        return X, Y, FES



# ============================================================
# 8. 强凸分布（标准各向同性高斯）
# ============================================================

class StronglyConvexGaussian(TargetDistribution):
    """
    强凸高斯分布（用于验证指数收敛）
    
    π(x) ∝ exp(-0.5 * ||x - μ||^2 / σ^2)
    
    标准各向同性高斯，LSI 常数 λ = 1/σ^2
    当 σ 较小时，λ 较大，指数收敛很快
    """
    
    def __init__(self, points, mean=(0, 0), sigma=1.0):
        super().__init__(points)
        self.mean = np.array(mean)
        self.sigma = sigma
        self.name = f"Strongly Convex Gaussian (σ={sigma})"
    
    def target_density(self, x, y):
        diff_x = x - self.mean[0]
        diff_y = y - self.mean[1]
        exponent = -0.5 * (diff_x**2 + diff_y**2) / self.sigma**2
        return np.exp(exponent)
    
    def compute(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        return self.target_density(x, y)
    
    def get_density_func(self):
        return self.target_density



# ============================================================

# ============================================================
# 工厂函数
# ============================================================

def get_target_distribution(name, points):
    """
    根据名称获取目标分布

    Parameters:
    -----------
    name : str
        分布名称，可选：
        - 'ring_double_well' : 环状双峰（原始）
        - 'gaussian_mixture' : 高斯混合 (8峰规则)
        - 'asymmetric_gaussian' : 非对称高斯混合 (7峰)
        - 'double_banana' : 双香蕉
        - 'gaussian' : 单峰高斯
        - 'double_gaussian' : 双峰高斯
        - 'anisotropic' : 各向异性高斯
        - 'single_ring' : 单环
        - 'four_peaks' : 四峰
        - 'muller_brown' : Müller-Brown 势能 (新增)
    points : array
        网格点坐标
    """
    targets = {
        'ring_double_well': RingDoubleWell(points),
        'gaussian_mixture': GaussianMixture(points),
        'asymmetric_gaussian': AsymmetricGaussianMixture(points),
        'double_banana': DoubleBanana(points),
        'gaussian': GaussianSinglePeak(points),
        'double_gaussian': DoubleGaussian(points),
        'anisotropic': AnisotropicGaussian(points),
        'single_ring': SingleRing(points),
        'four_peaks': FourPeaks(points),
        'muller_brown': MullerBrownTarget(points),  # 新增
        'strongly_convex': StronglyConvexGaussian(points),           # 新增
    }

    if name not in targets:
        raise ValueError(f"Unknown target: {name}. Available: {list(targets.keys())}")

    return targets[name]