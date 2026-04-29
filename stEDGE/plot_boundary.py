from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import seaborn as sns
from sklearn.decomposition import PCA

def prepare_interface_data(adata, belt_indices, gene_name):
    """提取界面带坐标和基因表达量"""
    coords = adata.obsm["spatial"][belt_indices]
    # 获取表达量 (处理稀疏矩阵)
    if hasattr(adata.raw.X, "toarray"):
        expr = adata[belt_indices, gene_name].X.toarray().flatten()
    else:
        expr = adata[belt_indices, gene_name].X.flatten()
        
    return coords, expr

def sort_points_by_main_axis(coords):
    """通过PCA对坐标进行初步排序，确定曲线的大致流向"""
    pca = PCA(n_components=1)
    projected_vals = pca.fit_transform(coords)
    sort_idx = np.argsort(projected_vals.flatten())
    return coords[sort_idx], sort_idx

def fit_interface_spline(sorted_coords, s=None, k=3):
    """
    s: 平滑参数，值越大越平滑；k: 样条阶数 (3代表三次样条)
    """
    x = sorted_coords[:, 0]
    y = sorted_coords[:, 1]
    
    # tck: 样条表示形式, u: 参数化坐标 (0到1)
    tck, u = interpolate.splprep([x, y], s=s, k=k)
    
    # 生成中轴线的精细坐标
    u_new = np.linspace(0, 1, 100)
    midline_coords = interpolate.splev(u_new, tck)
    
    return tck, np.array(midline_coords).T

def plot_cross_boundary_gradient(distances, expr, gene_name, label_a, label_b, save_path=None):
    import pandas as pd
    import seaborn as sns
    
    df_plot = pd.DataFrame({'Distance': distances, 'Expression': expr})

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    # 1. 绘制原始数据的极淡散点
    sns.scatterplot(data=df_plot, x='Distance', y='Expression', 
                    color='gray', s=6, alpha=0.1, edgecolor=None, ax=ax)

    # 2. 绘制跨界平滑拟合曲线
    # lowess=True 会根据局部数据拟合，最适合展示这种非线性的转换过程
    sns.regplot(data=df_plot, x='Distance', y='Expression', 
                scatter=False, lowess=True, color='#2878B5', 
                line_kws={"linewidth": 2.5, "alpha": 0.9}, ax=ax)

    # 3. 强调界面中心线 (x=0)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    
    # 4. 设置坐标轴与标签
    ax.set_title(f"$\it{{{gene_name}}}$ Expression Profile", fontsize=11, pad=12)
    ax.set_ylabel("Normalized Expression", fontsize=9)
    
    # 动态设置横坐标：指示方向
    ax.set_xlabel(f"Distance to Interface ($\mu m$)\n $\longleftarrow$ {label_a} | {label_b} $\longrightarrow$", fontsize=9)
    
    # 5. 极简风格美化
    sns.despine(trim=False)
    ax.tick_params(labelsize=8)
    ax.grid(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_interface_skeleton(adata, tck, midline_points, side_a, side_b, label_a, label_b, save_path=None):
    """
    专门用于可视化法线距离基准线（中轴线）的调试函数
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. 统一转换索引为坐标提取用的 mask
    n_obs = adata.n_obs
    mask_a = np.zeros(n_obs, dtype=bool)
    mask_b = np.zeros(n_obs, dtype=bool)
    
    idx_a = np.where(side_a)[0] if side_a.dtype == bool else side_a
    idx_b = np.where(side_b)[0] if side_b.dtype == bool else side_b
    mask_a[idx_a] = True
    mask_b[idx_b] = True
    
    coords_all = adata.obsm["spatial"]
    belt_mask = mask_a | mask_b

    # 2. 开始绘图
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # 绘制背景点（全体 Spot）
    ax.scatter(coords_all[:, 0], coords_all[:, 1], c='lightgray', s=1, alpha=0.1)

    # 绘制界面带的两侧点 (Side A 蓝色, Side B 橙色)
    ax.scatter(coords_all[mask_a, 0], coords_all[mask_a, 1], 
               c='#1f77b4', s=15, alpha=0.6, label=f'{label_a} (Dist < 0)')
    ax.scatter(coords_all[mask_b, 0], coords_all[mask_b, 1], 
               c='#ff7f0e', s=15, alpha=0.6, label=f'{label_b} (Dist > 0)')

    # 3. 绘制核心中轴线 (Spline)
    ax.plot(midline_points[:, 0], midline_points[:, 1], 
            color='black', linewidth=2.5, linestyle='-', label='Interface Baseline (0)', zorder=10)

    # 绘制 Spline 的起点和终点（辅助判断排序方向）
    ax.scatter(midline_points[0, 0], midline_points[0, 1], c='green', s=50, marker='D', label='Spline Start', zorder=11)
    ax.scatter(midline_points[-1, 0], midline_points[-1, 1], c='red', s=50, marker='X', label='Spline End', zorder=11)

    # 4. Y 轴镜像转置 (保持空间解剖一致性)
    ax.invert_yaxis()

    # 5. 图例与美化
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    ax.set_aspect('equal')
    ax.set_title(f"Debug: {label_a} | {label_b} Interface Skeleton", fontsize=12)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y (Inverted)")
    
    # 标注文字
    pos_a = coords_all[mask_a].mean(axis=0)
    pos_b = coords_all[mask_b].mean(axis=0)
    ax.text(pos_a[0], pos_a[1], label_a, weight='bold', color='#1f77b4', ha='center')
    ax.text(pos_b[0], pos_b[1], label_b, weight='bold', color='#ff7f0e', ha='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def get_interface_skeleton(adata, side_a, side_b, seeds_a, seeds_b, s_para=30):
    import numpy as np
    
    # 1. 统一转换为整数索引
    def to_idx(s):
        return np.where(s)[0] if (isinstance(s, np.ndarray) and s.dtype == bool) else s
    
    idx_side_a = to_idx(side_a)
    idx_side_b = to_idx(side_b)
    idx_seeds_a = to_idx(seeds_a)
    idx_seeds_b = to_idx(seeds_b)
    
    # 2. 计算用于拟合的 belt (只用 side_a 和 side_b，保证贴合度)
    belt_indices_global = np.union1d(idx_side_a, idx_side_b)
    # 确保提取的是当前传入 adata 的坐标
    fit_coords = adata.obsm["spatial"][belt_indices_global]
    
    # 3. 拟合逻辑 (严格使用 fit_coords)
    # 排序和拟合必须基于同样的点集
    sorted_coords, _ = sort_points_by_main_axis(fit_coords) 
    tck, midline_points = fit_interface_spline(sorted_coords, s=len(fit_coords) * s_para)
    
    # 4. 构建用于绘图的 adata_debug (包含 side 和 seeds)
    all_interface_indices = np.unique(np.concatenate([
        idx_side_a, idx_side_b, idx_seeds_a, idx_seeds_b
    ]))
    mask_debug = np.isin(np.arange(adata.n_obs), all_interface_indices)
    adata_debug = adata[mask_debug].copy()
    
    # 5. 映射子集内的 side 掩码用于绘图颜色区分
    # 这里使用 global 索引去匹配子集的 obs_names
    local_side_a = np.isin(all_interface_indices, idx_side_a)
    local_side_b = np.isin(all_interface_indices, idx_side_b)
    
    # 6. 调用绘图
    plot_interface_skeleton(
        adata_debug, tck, midline_points, 
        side_a=local_side_a, 
        side_b=local_side_b, 
        label_a="Side A", label_b="Side B"
    )
    
    return tck, midline_points, belt_indices_global



def plot_gene_interface_gradient(adata, tck, side_a, side_b, seeds_a, seeds_b, gene_name, label_a, label_b, save_path=None):
    """
    第二部分：基于子集聚焦，绘制特定基因的界面梯度图
    """
    import numpy as np
    from scipy import interpolate

    # 1. 内部处理：构建局部子集 (adata_debug)
    # 合并所有界面相关的 spot 索引
    all_interface_indices = np.unique(np.concatenate([
        np.where(side_a)[0] if side_a.dtype == bool else side_a,
        np.where(side_b)[0] if side_b.dtype == bool else side_b,
        np.where(seeds_a)[0] if seeds_a.dtype == bool else seeds_a,
        np.where(seeds_b)[0] if seeds_b.dtype == bool else seeds_b
    ]))
    
    # 转换为布尔掩码并提取子集
    mask_debug = np.isin(np.arange(adata.n_obs), all_interface_indices)
    adata_debug = adata[mask_debug].copy()
    
    # 2. 重新确定子集内的 side_a 掩码 (用于距离的正负号判定)
    # 我们需要在 adata_debug 的坐标系下知道哪些是 A 侧
    idx_a_global = np.where(side_a)[0] if side_a.dtype == bool else side_a
    # 映射到 adata_debug 的局部布尔掩码
    local_mask_a = np.isin(all_interface_indices, idx_a_global)

    # 3. 提取表达数据与坐标 (基于子集)
    # 注意：这里的 coords 对应的是 adata_debug 里的所有点
    coords = adata_debug.obsm["spatial"]
    
    if hasattr(adata_debug[:, gene_name].X, "toarray"):
        expr = adata_debug[:, gene_name].X.toarray().flatten()
    else:
        expr = adata_debug[:, gene_name].X.flatten()

    # 4. 基于传入的 tck 生成中轴线精细点
    u_fine = np.linspace(0, 1, 2000)
    line_points = np.array(interpolate.splev(u_fine, tck)).T

    # 5. 计算带符号的正交距离 (Signed Normal Distance)
    signed_distances = []
    for i, point in enumerate(coords):
        # 计算该点到中轴线上所有点的最短距离
        dists = np.linalg.norm(line_points - point, axis=1)
        min_dist = np.min(dists)
        
        # 使用 local_mask_a 判定：A侧为负，B侧为正
        val = -min_dist if local_mask_a[i] else min_dist
        signed_distances.append(val)
    
    signed_distances = np.array(signed_distances)

    # 6. 调用绘图函数
    # 此时绘图只会包含 mask_debug 选中的这些 spot
    plot_cross_boundary_gradient(
        signed_distances, expr, gene_name, label_a, label_b, save_path=save_path
    )

    return signed_distances, expr

