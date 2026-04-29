import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import davies_bouldin_score
import stEDGE
import scanpy as sc
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import to_rgb
import pandas as pd
import anndata as ad
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from . import TGL_module


def plot_spatial(
    adata,
    colors,
    titles=None,
    spot_size=10,
    cmap="inferno",
    figsize=(5, 5),
    alpha=1,
    frameon=False,
    legend_loc=None,
    colorbar_loc="right",
    save_path=None,
    show=False,
    dpi=600,
):
    """可复用的 scanpy spatial 绘图封装（支持单图/多 panel）。"""
    plt.figure(figsize=figsize)
    sc.pl.spatial(
        adata,
        color=colors,
        cmap=cmap,
        title=titles if titles is not None else colors,
        alpha=alpha,
        frameon=frameon,
        legend_loc=legend_loc,
        colorbar_loc=colorbar_loc,
        spot_size=spot_size,
        show=False,
    )


    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)

    # if show:
    #     plt.show()
    # else:
    #     plt.close()


def plot_p_edge_distribution(adata, output_dir, bins=50, color='skyblue', show_median=False):
    """
    绘制 adata.obs['p_edge'] 的直方图，并保存图像。

    参数:
    - adata: AnnData
        包含 'p_edge' 列的 AnnData 对象。
    - output_path: str
        图像保存路径，如 'output/p_edge_value.png'。
    - bins: int
        直方图的柱子数量（默认 50）。
    - color: str
        柱子颜色（默认 skyblue）。
    - show_median: bool
        是否显示中位数分割线（默认 False）。
    """

    p_edge = adata.obs['p_edge'].values

    plt.figure(figsize=(8, 4))
    plt.hist(p_edge, bins=bins, color=color, edgecolor='k', alpha=0.7)

    if show_median:
        median = np.median(p_edge)
        plt.axvline(median, color='red', linestyle='--', label=f'Median: {median:.3f}')
        plt.legend()

    plt.xlabel("p_edge")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()

    if output_dir:
        outpath = os.path.join(output_dir, f"p_edge_value.png")
        plt.savefig(outpath, dpi=600, bbox_inches="tight", pad_inches=0.1)


def plot_gradient_field(adata,
                        output_dir,
                        step=1,
                        scale=150,
                        flip_y=True,
                        show_points=True,
                        point_alpha=0.3,
                        point_size=10,
                        cmap='RdBu_r',
                        figsize=(5, 5),
                        dpi=600):

    coords = adata.obsm['spatial']
    p_edge = adata.obs['p_edge'].values
    grad_p_edge = adata.obsm['grad_p_edge']

    idx = np.arange(0, len(coords), step)

    x = coords[:, 0]
    y = -coords[:, 1] if flip_y else coords[:, 1]

    gx = grad_p_edge[:, 0]
    gy = grad_p_edge[:, 1] if not flip_y else -grad_p_edge[:, 1]

    # ✅ 计算 magnitude
    grad_mag = np.sqrt(gx**2 + gy**2)

    # ✅ 单位向量归一化（避免除零）
    eps = 1e-9
    gx_norm = gx / (grad_mag + eps)
    gy_norm = gy / (grad_mag + eps)

    plt.figure(figsize=figsize)

    if show_points:
        plt.scatter(x, y, c=p_edge, cmap='gray', s=point_size, alpha=point_alpha)

    plt.quiver(
        x[idx],
        y[idx],
        gx_norm[idx],      # ✅ 用归一化后的向量
        gy_norm[idx],
        grad_mag[idx],     # ✅ 仍然用强度上色（可选）
        cmap=cmap,
        scale=scale,
        headwidth=5,
        width=0.003
    )

    plt.colorbar(label='Gradient magnitude')
    plt.title("Gradient Field")
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()

    if output_dir:
        outpath = os.path.join(output_dir, f"Gradient_Field.png")
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight", pad_inches=0.1)

    return


def plot_gradient_magnitude(adata,
                            output_dir,
                            flip_y=True,
                            point_size=5,
                            figsize=(5, 5),
                            dpi=600,
                            cmap='viridis'):
    """
    绘制 p_edge 的梯度模长（Gradient Magnitude）散点图。

    参数:
    - coords: ndarray (N, 2)
        空间坐标。
    - grad_p_edge: ndarray (N, 2)
        梯度向量。
    - output_path: str
        图像保存路径，例如 "output/grad_magnitude.png"。
    - flip_y: bool, 默认 True
        是否翻转 y 轴（常用于空间转录组数据）。
    - point_size: int
        散点大小。
    - figsize: tuple
        图像大小。
    - dpi: int
        图像分辨率。
    - cmap: str
        色图名称。
    """
    coords = adata.obsm['spatial']
    grad_p_edge = adata.obsm['grad_p_edge']
    
    grad_magnitude = np.linalg.norm(grad_p_edge, axis=1)
    x = coords[:, 0]
    y = -coords[:, 1] if flip_y else coords[:, 1]

    plt.figure(figsize=figsize)
    plt.scatter(x, y, c=grad_magnitude, cmap=cmap, s=point_size)
    plt.colorbar(label='|grad_p_edge|')
    plt.title('Gradient Magnitude of p_edge')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    if output_dir:
        outpath = os.path.join(output_dir, f"Gradient Field.png")
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    return


def plot_upper_bound_tracking(output_csv_path,
                              output_dir,
                              seed_range=(1, 200),
                              min_points=10,
                              max_steps=600,
                              smooth_factor=1,
                              figsize=(14, 6),
                              dpi=600):
    """
    可视化 upper_bound_tracking.csv 中不同种子 seed_label 的上界轨迹（平滑插值）。

    参数:
    - csv_path: str
        CSV 文件路径。
    - output_path: str
        输出图像路径。
    - seed_range: tuple(int, int)
        要可视化的种子标签范围 (起始, 结束)，如 (1, 200)。
    - min_points: int
        每个种子最少点数，低于该值将不进行平滑显示。
    - max_steps: int
        每个种子轨迹最多显示的点数。
    - smooth_factor: float
        插值平滑因子（越大越平滑）。
    - figsize: tuple
        图像尺寸。
    - dpi: int
        输出图像分辨率。
    """
    df = pd.read_csv(output_csv_path)
    df_filtered = df[(df["seed_label"] >= seed_range[0]) & (df["seed_label"] <= seed_range[1])]

    plt.figure(figsize=figsize)

    for seed_label, group in df_filtered.groupby("seed_label"):
        group = group.reset_index(drop=True)
        if len(group) > min_points:
            y = group["upper_bound"].values[:max_steps]
            x = np.arange(len(y))
            spline = UnivariateSpline(x, y, s=smooth_factor)
            x_dense = np.linspace(0, len(y) - 1, max_steps)
            y_smooth = spline(x_dense)
            plt.plot(x_dense, y_smooth, linewidth=1, label=f"Seed {seed_label}")

    plt.xlabel("Step")
    plt.ylabel("Upper Bound")
    plt.title("Upper Bound Tracking of Seeds")
    plt.tight_layout()
    if output_dir:
        outpath = os.path.join(output_dir, f"upper_bound_tracking_{max_steps}.png")
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    return


def plot_umap_with_db_index(adata,
                             label_key='domain_merge',
                             umap_key='X_umap',
                             output_path='stEDGE_UMAP.png',
                             dpi=1000,
                             figsize=(5, 5),
                             s=30,
                             outline_width=(0.1, 0.05),
                             legend_fontsize=10):
    """
    绘制带有 DB 指数的 UMAP 图。

    参数：
    - adata: AnnData
        含有 UMAP 和聚类标签的 AnnData 对象。
    - label_key: str
        聚类标签所在 adata.obs 的列名。
    - umap_key: str
        UMAP 坐标所在 adata.obsm 的键。
    - output_path: str
        图像保存路径。
    - dpi: int
        输出图像分辨率。
    - figsize: tuple
        图像尺寸。
    - s: int
        每个点的大小。
    - outline_width: tuple
        点轮廓宽度。
    - legend_fontsize: int
        图例字体大小。
    """

    # 计算 DB 指数
    X = adata.obsm[umap_key]
    labels = adata.obs[label_key].values
    db_index = davies_bouldin_score(X, labels)

    # 设置图像大小
    plt.rcParams["figure.figsize"] = figsize

    # 绘图
    sc.pl.umap(
        adata,
        color=label_key,
        title=[f"UMAP colored by {label_key} (DB: {db_index:.3f})"],
        legend_fontoutline=1,
        add_outline=True,
        s=s,
        outline_width=outline_width,
        legend_fontsize=legend_fontsize,
        frameon=False,
        show=False
    )

    # 保存
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()



def plot_top_genes_per_domain_combined(adata, group_key='domain', top_n=5, spot_size=350, out_dir=None):
    os.makedirs(out_dir, exist_ok=True)

    # 提取差异基因结果为 DataFrame
    markers = sc.get.rank_genes_groups_df(adata, group=None, key='rank_genes_groups')
    top_markers = markers.groupby('group').head(top_n)
    domains = top_markers['group'].unique()

    for domain in domains:
        genes = top_markers[top_markers['group'] == domain]['names'].tolist()
        domain_str = str(domain)

        # 创建子图：1 行（域图 + top_n 基因图）共 top_n + 1 个子图
        fig, axes = plt.subplots(1, top_n + 1, figsize=(3 * (top_n + 1), 4))

        # ------- 子图 1：该域在组织中的位置 -------
        sc.pl.spatial(
            adata,
            color=group_key,
            groups=[domain],
            spot_size=spot_size,
            ax=axes[0],
            show=False,
            title=f"Domain {domain_str}"
        )

        # ------- 子图 2~n+1：top N 个 marker genes -------
        for i, gene in enumerate(genes):
            sc.pl.spatial(
                adata,
                color=gene,
                spot_size= spot_size,
                ax=axes[i + 1],
                show=False,
                title=gene
            )

        # ------- 保存合并图 -------
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"domain_{domain_str}_combined.png"), dpi=600)
        plt.close(fig)  # 更清晰关闭 figure，避免图太多警告

    print(f"所有域的合并图已保存至：{out_dir}")



def plot_domain_with_alpha_series(adata, output_dir, cmap='inferno', spot_size=350):
    """
    绘制 domain 图像，alpha 从 0 到 1（步长 0.1），共保存 11 张图到 output_dir。

    Parameters:
    - adata: AnnData 对象
    - output_dir: 保存图像的文件夹路径
    - cmap: 颜色映射（默认为 inferno）
    - spot_size: 点大小（默认 350）
    """
    os.makedirs(output_dir, exist_ok=True)

    for alpha in np.arange(0, 1.01, 0.1):
        plt.rcParams["figure.figsize"] = (10, 10)
        sc.pl.spatial(
            adata,
            color="domain",
            cmap=cmap,
            title=[f"alpha = {alpha:.1f}"],
            alpha=alpha,
            frameon=False,
            colorbar_loc=None,
            spot_size=spot_size,
            show=False
        )
        output_path = os.path.join(output_dir, f"domain_alpha_{alpha:.1f}.png")
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def plot_clusters_results_spatial(adata, output_dir, spot_size=250):
    """
    可视化 adata.obsm["clusters_results"] 的每一列结果，并保存图片
    """
    if "clusters_results" not in adata.obsm:
        raise ValueError("adata.obsm 中没有 'clusters_results'")
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm 中没有 'spatial'")
    
    os.makedirs(output_dir, exist_ok=True)

    clusters_results = pd.DataFrame(
        adata.obsm["clusters_results"], 
        index=adata.obs_names
    )

    for col in clusters_results.columns:
        # 临时存入 obs，方便 sc.pl.spatial 调用
        adata.obs[f"cluster_{col}"] = clusters_results[col].astype("category")
        
        plt.rcParams["figure.figsize"] = (8, 8)
        sc.pl.spatial(
            adata,
            color=f"cluster_{col}",
            cmap="inferno",
            title=[f"Cluster {col}"],
            alpha=1,
            frameon=False,
            colorbar_loc=None,
            spot_size=spot_size,
            show=False
        )
        outpath = os.path.join(output_dir, f"cluster_{col}.png")
        plt.savefig(outpath, dpi=600, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    print(f"✅ 所有聚类结果可视化完成，已保存至: {output_dir}")



import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import os

def batch_detect_and_plot(
    adata,
    output_process_dir,
    spot_size=20,
    min_size=20,
    th_range=np.arange(0.1, 0.91, 0.1),
    con_region_key="con_region_ori"
):
    """
    遍历 th_inner 阈值，调用 detect_new_domains_from_unassigned，
    并保存对应的空间可视化结果。
    
    Parameters
    ----------
    adata : AnnData
        输入 AnnData 对象
    output_process_dir : str
        输出目录
    spot_size : int
        空间可视化点大小
    min_size : int
        新域的最小连通分量大小
    th_range : iterable
        遍历的 th_inner 值范围 (默认 0.1 ~ 0.9, 步长 0.1)
    con_region_key : str
        adata.obs 中用于存储原始 con_region 的 key
    """

    out_dir = os.path.join(output_process_dir, "th_inner")
    os.makedirs(out_dir, exist_ok=True)

    for th_inner in th_range:
        # 检测新域
        stEDGE.utils.detect_new_domains_from_unassigned(
            adata,
            th_inner=th_inner,
            min_size=min_size,
            con_region_key=con_region_key
        )

        # 绘制结果
        plt.rcParams["figure.figsize"] = (8, 8)
        sc.pl.spatial(
            adata,
            color="con_region",
            cmap="inferno",
            title=[f"th_inner_{th_inner:.1f}"],
            alpha=1,
            frameon=False,
            colorbar_loc=None,
            spot_size=spot_size,
            show=False
        )
        # 保存图片
        plt.savefig(
            f"{out_dir}/con_region_{th_inner:.1f}.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.1
        )
        plt.close()


def plot_coarse_scan(R_sorted, metrics, similarities, intervals=None):
    """
    可视化粗扫描结果：簇数变化 & 相邻分辨率相似度

    Parameters
    ----------
    R_sorted : list of str
        分辨率标签列表，例如 ['leiden_1.00', 'leiden_1.10', ...]
    metrics : dict
        每个分辨率对应的 {'labels': ..., 'k': ...}
    similarities : dict
        相邻分辨率的相似度结果
        例如 {('leiden_2.90', 'leiden_3.00'): {'ARI': 0.89, 'Δk': 4}, ...}
    intervals : tuple or list of tuples, optional
        需要高亮的稳定区间，可以是单个区间 (start, end)
        或多个区间 [(s1, e1), (s2, e2), ...]
    """

    # --- 提取分辨率值 ---
    def parse_res(label):
        return float(label.split("_")[1])

    R_vals = [parse_res(r) for r in R_sorted]
    ks = [metrics[r]['k'] for r in R_sorted]

    # --- 处理 intervals 参数 ---
    highlight_ranges = []
    if intervals is not None:
        if isinstance(intervals, tuple) and len(intervals) == 2:
            highlight_ranges = [intervals]
        elif isinstance(intervals, list):
            highlight_ranges = intervals
        else:
            raise ValueError("intervals 必须是 (start, end) 或 [(s1, e1), (s2, e2), ...]")

    plt.figure(figsize=(14, 6))

    # 1️⃣ 簇数随分辨率变化
    plt.subplot(2, 1, 1)
    plt.plot(R_vals, ks, marker='o', label="All resolutions")

    # 高亮指定区间的点
    for (start, end) in highlight_ranges:
        mask = [(start <= r <= end) for r in R_vals]
        plt.scatter(
            [r for r, m in zip(R_vals, mask) if m],
            [k for k, m in zip(ks, mask) if m],
            color="red", s=40, zorder=5, label=f"Selected [{start}, {end}]"
        )

    plt.xlabel("Resolution r")
    plt.ylabel("Number of clusters k(r)")
    plt.title("Cluster count vs resolution")
    for r, k in zip(R_vals, ks):
        plt.text(r, k, str(k), fontsize=8, ha='center', va='bottom')

    # plt.legend()

    # 2️⃣ 相邻分辨率 ARI
    plt.subplot(2, 1, 2)
    R_pairs = [f"{parse_res(r1):.2f}-{parse_res(r2):.2f}" for (r1, r2) in similarities.keys()]
    ARIs = [v['ARI'] for v in similarities.values()]
    Δks = [v['Δk'] for v in similarities.values()]

    plt.plot(range(len(R_pairs)), ARIs, marker='o', label="ARI (similarity)")

    # 高亮区间对应的点（相邻分辨率落在区间内的）
    for (start, end) in highlight_ranges:
        mask = [
            (start <= parse_res(r1) <= end) and (start <= parse_res(r2) <= end)
            for (r1, r2) in similarities.keys()
        ]
        plt.scatter(
            [i for i, m in enumerate(mask) if m],
            [ari for ari, m in zip(ARIs, mask) if m],
            color="red", s=40, zorder=5, label=f"Selected [{start}, {end}]"
        )

    plt.xticks(range(len(R_pairs)), R_pairs, rotation=45, ha='right')
    plt.xlabel("Resolution intervals")
    plt.ylabel("ARI")
    plt.ylim(0, 1.05)
    plt.title("Stability between adjacent resolutions")

    # 标出 Δk
    for i, (ari, dk) in enumerate(zip(ARIs, Δks)):
        plt.text(i, ari, f"Δk={dk}", fontsize=7, ha='center', va='bottom')

    # plt.legend()
    plt.tight_layout()
    plt.show()


def plot_domain_graph(
    adata,
    domain_key: str = "domain",
    color_mode: str = "dti",           # "dti" 或 "cluster"
    min_sim_threshold: float = 0.1,
    max_edge_width: float = 8.0,
    node_size_base: float = 80,
    node_size_scale: float = 100,
    figsize=(8, 6),
    legend=False,
    output_dir=None,
):
    """
    画一个 domain graph：
      - 每个点 = 一个 domain
      - 节点大小 ∝ 域内 spot 数量
      - color_mode="dti"     : 节点颜色 = DTI（连续 colormap）
      - color_mode="cluster" : 节点颜色 = coarse 域/簇颜色（来自 adata.uns[domain_key + "_coarse_colors"]）
      - 边粗细 ∝ domain 相似度（来自 adata.uns[f"{domain_key}_similarity"]）
    """

    # ---------- 1. 相似度矩阵 ----------
    similarity_key = f"{domain_key}_similarity"
    if similarity_key not in adata.uns:
        raise KeyError(f"{similarity_key} 不在 adata.uns 中，请先构建 domain 相似度矩阵。")

    sim_df = adata.uns[similarity_key].copy()
    sim_df.index = sim_df.index.astype(str)
    sim_df.columns = sim_df.columns.astype(str)

    domains = sim_df.index.to_list()
    K = len(domains)
    if K == 0:
        print("No domains to plot.")
        return

    # ---------- 2. 域大小（节点尺寸） ----------
    dom_series = adata.obs[domain_key].astype(str)
    size_counts = dom_series.value_counts()
    domain_sizes = np.array([size_counts.get(d, 0) for d in domains], dtype=float)

    if domain_sizes.max() > domain_sizes.min():
        size_norm = (domain_sizes - domain_sizes.min()) / (domain_sizes.max() - domain_sizes.min())
    else:
        size_norm = np.zeros_like(domain_sizes)
    node_sizes = node_size_base + node_size_scale * size_norm

    # ---------- 3. DTI 信息（仅在 color_mode="dti" 时使用） ----------
    dti_series = None
    dti_key = f"{domain_key}_DTI"
    if dti_key in adata.obs.columns:
        dti_series = adata.obs.groupby(domain_key, observed=True)[dti_key].first()
    if dti_series is not None:
        dti_series = dti_series.copy()
        dti_series.index = dti_series.index.astype(str)

    # ---------- 4. cluster 信息（仅在 color_mode="cluster" 时使用） ----------
    cluster_key = f"{domain_key}_coarse"  # 约定：coarse 域标签列名
    coarse_color_key = f"{domain_key}_coarse_colors"  # 约定：颜色列表存放位置

    cluster_series = None
    if color_mode.lower() == "cluster":
        if cluster_key not in adata.obs.columns:
            raise KeyError(
                f"color_mode='cluster'，但在 adata.obs 中找不到列 '{cluster_key}'，"
                f"请先在 obs 中生成 {cluster_key}（coarse 域 / 域簇标签）。"
            )

        # cluster 的 category & 颜色列表
        cluster_series = adata.obs[cluster_key].astype("category")
        cluster_cats = cluster_series.cat.categories.astype(str)

        if coarse_color_key in adata.uns:
            # 方案 A: 使用预设颜色 (优先)
            palette = list(adata.uns[coarse_color_key])
            print(f"use adata.uns['{coarse_color_key}'] for cluster colors.")
        else:
            # 方案 B: 回退到 tab20 颜色
            print(f"警告：adata.uns 中未找到 '{coarse_color_key}'，回退使用 Matplotlib 的 'tab20' 颜色。")
            
            # 从 Matplotlib 获取 tab20 颜色列表（最多 20 种颜色）
            cmap = plt.colormaps.get_cmap("tab20")
            # 转换为十六进制颜色代码
            palette = [mcolors.to_hex(cmap(i % cmap.N)) for i in range(len(cluster_cats))]
            adata.uns[coarse_color_key] = palette  # 存回 adata.uns 以备后用


        # 如果颜色数少于类别数，重复补齐
        if len(palette) < len(cluster_cats):
            times = len(cluster_cats) // len(palette) + 1
            palette = (palette * times)[: len(cluster_cats)]

        cluster_color_map = dict(zip(cluster_cats, palette))


        # domain -> cluster 映射：每个 domain 取其内部第一个 coarse 标签
        dom2cl = (
            adata.obs[[domain_key, cluster_key]]
            .astype({domain_key: str})
            .groupby(domain_key)[cluster_key]
            .first()
        )
        dom2cl.index = dom2cl.index.astype(str)  # 确保 index 是 str

    # ---------- 5. 构图 ----------
    G = nx.Graph()
    for d in domains:
        d_str = str(d)
        dti_val = (
            float(dti_series.get(d_str))
            if dti_series is not None and d_str in dti_series.index
            else np.nan
        )
        cl_val = (
            dom2cl.get(d_str)
            if (color_mode.lower() == "cluster" and 'dom2cl' in locals() and d_str in dom2cl.index)
            else np.nan
        )
        G.add_node(d_str, DTI=dti_val, CLUSTER=cl_val)

    # 添加边
    sim = sim_df.to_numpy()
    upper = sim[np.triu_indices(K, k=1)]
    if upper.size == 0:
        print("No domain pairs to connect.")
        return

    for i in range(K):
        for j in range(i + 1, K):
            w = sim[i, j]
            if w < min_sim_threshold:
                continue
            G.add_edge(str(domains[i]), str(domains[j]), weight=float(w))

    if G.number_of_edges() == 0:
        print("All similarities below threshold; no edges to draw.")
        return

    # ---------- 6. 布局 ----------
    pos = nx.spring_layout(G, seed=0, k=0.3, iterations=50)
    nodes_list = list(G.nodes())

    # ---------- 7. 节点颜色 ----------
    if color_mode.lower() == "dti":
        if dti_series is None:
            raise ValueError("color_mode='dti' 但找不到任何 DTI 信息，请检查 obs 中是否存在相关列。")

        dti_vals = np.array([G.nodes[d]["DTI"] for d in nodes_list], dtype=float)
        if np.isnan(dti_vals).any():
            mask = ~np.isnan(dti_vals)
            mean_val = np.nanmean(dti_vals[mask]) if mask.any() else 0.5
            dti_vals[np.isnan(dti_vals)] = mean_val

        dti_min, dti_max = dti_vals.min(), dti_vals.max()
        if dti_max > dti_min:
            node_color_vals_num = (dti_vals - dti_min) / (dti_max - dti_min)
        else:
            node_color_vals_num = np.zeros_like(dti_vals)

        cmap = cm.get_cmap("inferno")
        show_colorbar = True
        colorbar_label = "DTI (normalized)"

    elif color_mode.lower() == "cluster":
        # 直接使用 adata.uns[domain_key + "_coarse_colors"] 中的颜色
        cl_vals = np.array([G.nodes[d]["CLUSTER"] for d in nodes_list], dtype=object)
        cl_vals_str = pd.Series(cl_vals, dtype="category").astype(str).values
        node_color_vals_cat = [cluster_color_map.get(c, "#808080") for c in cl_vals_str]

        show_colorbar = False
        colorbar_label = None

    else:
        raise ValueError("color_mode 必须是 'dti' 或 'cluster'。")

    # ---------- 8. 边粗细 ----------
    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    ew_min, ew_max = edge_weights.min(), edge_weights.max()
    if ew_max > ew_min:
        edge_widths = max_edge_width * (edge_weights - ew_min) / (ew_max - ew_min)
    else:
        edge_widths = np.full_like(edge_weights, max_edge_width / 2)

    # ---------- 9. 画图 ----------
    plt.figure(figsize=figsize)

    if color_mode.lower() == "dti":
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_color_vals_num,
            cmap=cmap,
            alpha=0.8,
        )
        rgba = cmap(node_color_vals_num)
    else:  # cluster: 直接用 hex 颜色
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_color_vals_cat,
            alpha=0.8,
        )
        rgb = np.array([to_rgb(c) for c in node_color_vals_cat])
        rgba = np.concatenate([rgb, np.ones((rgb.shape[0], 1))], axis=1)

    edges = nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.8,
    )

    # ---- 自适应标签颜色 ----
    luminance = 0.299 * rgba[:, 0] + 0.587 * rgba[:, 1] + 0.114 * rgba[:, 2]
    domain_to_idx = {d: i for i, d in enumerate(nodes_list)}

    for node, (x, y) in pos.items():
        idx = domain_to_idx[node]
        text_color = "white" if luminance[idx] < 0.5 else "black"
        plt.text(
            x, y, str(node),
            fontsize=7,
            ha="center", va="center",
            color=text_color,
        )

    if show_colorbar:
        cbar = plt.colorbar(nodes)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    plt.axis("off")

    # ---------- 10. 处理图例 ----------
    if legend:
        if color_mode.lower() == "cluster":
            from matplotlib.lines import Line2D
            # 构造图例句柄
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', 
                       label=cat,
                       markerfacecolor=cluster_color_map.get(cat, "#808080"), 
                       markersize=12)
                for cat in cluster_cats
            ]
            # 添加图例
            plt.legend(handles=legend_elements, 
                       title="Coarse Cluster", 
                       loc='center left', 
                       bbox_to_anchor=(1, 0.5)) # 放在图外，防止遮挡

    
    if color_mode.lower() == "dti":
        plt.title("Domain graph (node color = DTI)", fontsize=12)
    else:
        plt.title(f"Domain graph (node color = {domain_key}_coarse)", fontsize=12)

    plt.tight_layout()
    # plt.show()
    if output_dir:
        outpath = os.path.join(output_dir, f"{domain_key}_graph_{color_mode}.png")
        plt.savefig(outpath, dpi=600, bbox_inches="tight", pad_inches=0.1)

def plot_domain_boundary_graph(
    adata,
    figsize=(8, 6),
    node_size=300,
    domain_key="fine",
    title="Domain boundary graph",
    output_dir=None,
):
    min_width = 1.0
    max_width = 3.0
    cmap = "viridis"
    node_color = "skyblue"
    node_alpha = 0.9
    edge_alpha = 0.8
    k = 0.2
    iterations = 1000
    dpi = 600

    boundary_strength = adata.uns[f"{domain_key}_boundary_strength"].copy()

    # 1. 创建无向图
    G = nx.Graph()

    # 2. 添加边和边界强度作为权重
    for _, row in boundary_strength.iterrows():
        d1_str = str(row['domain_a'])
        d2_str = str(row['domain_b'])
        strength = float(row['strength'])
        
        G.add_edge(d1_str, d2_str, weight=float(strength))

    if G.number_of_edges() == 0:
        print("boundary_strength 中没有任何边可以画。")
        return

    # 提取边的权重 (即边界强度)
    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)

    # 归一化权重到 [min_width, max_width]
    w_min, w_max = edge_weights.min(), edge_weights.max()
    if w_max > w_min:
        widths = min_width + (edge_weights - w_min) / (w_max - w_min) * (max_width - min_width)
    else:
        # 所有边权一样，统一一个中间值
        widths = np.full_like(edge_weights, (min_width + max_width) / 2)

    # 3. 布局（固定 spring_layout）
    pos = nx.spring_layout(G, k=k, iterations=iterations, weight="weight")

    # 4. 绘图
    plt.figure(figsize=figsize, dpi=dpi)

    # 节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_color,
        alpha=node_alpha,
    )

    # 边：用 boundary_strength 映射到宽度和颜色
    edges = nx.draw_networkx_edges(
        G, pos,
        width=widths,
        edge_color=edge_weights,
        edge_cmap=plt.cm.get_cmap(cmap),
        alpha=edge_alpha,
    )

    # 标签
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="black")

    # colorbar 表示边界强度
    # cbar = plt.colorbar(edges)
    # cbar.set_label("Boundary strength")

    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    # plt.show()
    if output_dir:
        outpath = os.path.join(output_dir, f"domain_boundary.png")
        plt.savefig(outpath, dpi=600, bbox_inches="tight", pad_inches=0.1)

from matplotlib.ticker import MaxNLocator
def plot_domain_gain_log(
    adata: ad.AnnData,
    uns_key: str = "domain_gain_log",
    gain_key: str = "best_gain",
    title: str = "Best Gain per Domain Merge Step",
    ylabel: str = "Best Gain",
    figsize: tuple = (8, 4),
    output_dir: str = None
):
    # 1. 数据校验
    if uns_key not in adata.uns:
        raise KeyError(f"adata.uns 中缺少键: '{uns_key}'")
    
    df = adata.uns[uns_key]
    if gain_key not in df.columns:
        raise ValueError(f"DataFrame 中缺少增益列: '{gain_key}'")

    # 2. 绘图
    plt.figure(figsize=figsize)
    # 假设索引或默认列名为 'step'，如果没有 'step' 列，则使用 index 绘图
    x_data = df["step"] if "step" in df.columns else df.index

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.plot(x_data, df[gain_key], marker='o', linestyle='-', color='#1f77b4', markersize=4)
    
    plt.title(title)
    plt.xlabel("Merge Step")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f"{gain_key}_log.png")
        plt.savefig(outpath, dpi=600, bbox_inches="tight", pad_inches=0.1)
        print(f">>> Plot saved to: {outpath}")


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def plot_domain_tree(
    adata,
    figsize=(12, 6),
    fontsize=10,
    cmap_name='inferno',
    output_dir=None,
    dpi=600,
):
    """
    三层树（标签在节点内部）：
      Level 0: Coarse
      Level 1: Domain (Normal)
      Level 2: Fine (fine)

    边颜色 = 子节点 DTI：
      Coarse->Domain  用 domain 聚合 DTI
      Domain->Fine    用 fine 聚合 DTI
    """
    domain_map_key="domain_map"
    coarse_map_key="domain_coarse_map"
    fine_dti_col="fine_DTI"
    domain_dti_col="domain_DTI"
    fine_col="fine"
    domain_col="domain"

    node_size_base=220

    gap_fine_within_domain=0
    gap_between_domains=0.25
    gap_between_coarse=0.5
    edge_alpha=1

    # colormap
    show_colorbar=True

    if domain_map_key not in adata.uns or coarse_map_key not in adata.uns:
        raise KeyError("请确保 adata.uns 中包含 domain_map 和 domain_coarse_map（或你指定的 key）。")
    if fine_dti_col not in adata.obs:
        raise KeyError(f"adata.obs 缺少列：{fine_dti_col}")
    if domain_dti_col not in adata.obs:
        raise KeyError(f"adata.obs 缺少列：{domain_dti_col}")
    if fine_col not in adata.obs:
        raise KeyError(f"adata.obs 缺少列：{fine_col}")

    domain_map = adata.uns[domain_map_key]   # domain -> [fine]
    coarse_map = adata.uns[coarse_map_key]   # coarse -> [domain]

    # -------- 0) 统一成 str id（避免 1 vs "1"） ----------
    def _s(x): return str(x)

    coarse_ids = sorted([_s(c) for c in coarse_map.keys()], key=lambda x: int(x) if x.isdigit() else x)
    normal_ids = sorted({_s(n) for normals in coarse_map.values() for n in normals},
                        key=lambda x: int(x) if x.isdigit() else x)
    fine_ids = sorted({_s(f) for fines in domain_map.values() for f in fines},
                      key=lambda x: int(x) if x.isdigit() else x)

    # -------- 1) spot 层级标签（domain 可能不存在则推断） ----------
    region_series = adata.obs[fine_col].astype(str)

    if domain_col in adata.obs:
        domain_series = adata.obs[domain_col].astype(str)
    else:
        # 用 domain_map 反推：fine -> domain
        region_to_domain = {}
        for d, regs in domain_map.items():
            d = _s(d)
            for r in regs:
                r = _s(r)
                if r in region_to_domain and region_to_domain[r] != d:
                    raise ValueError(f"fine {r} 同时属于 domain {region_to_domain[r]} 和 {d}")
                region_to_domain[r] = d
        domain_series = region_series.map(region_to_domain).astype(str)

    # -------- 2) 计算每个 domain / fine 的聚合 DTI ----------
    region_dti = adata.obs.groupby(region_series)[fine_dti_col].mean()
    domain_dti = adata.obs.groupby(domain_series)[domain_dti_col].mean()

    # 转成 dict，key=str(id)
    region_dti = region_dti.to_dict()
    domain_dti = domain_dti.to_dict()

    # -------- 3) 收集节点 ----------
    nodes = {}
    for c in coarse_ids:
        nodes[f"C|{c}"] = {"label": c, "level": 0}
    for n in normal_ids:
        nodes[f"D|{n}"] = {"label": n, "level": 1}
    for f in fine_ids:
        nodes[f"F|{f}"] = {"label": f, "level": 2}

    # -------- 4) 构建边 + 记录每条边的 DTI（用于上色） ----------
    edges = []
    edge_value = {}  # (u,v) -> float or np.nan

    # Coarse -> Domain（用 Domain DTI）
    for c, normals in coarse_map.items():
        c = _s(c)
        u = f"C|{c}"
        for n in normals:
            n = _s(n)
            v = f"D|{n}"
            edges.append((u, v))
            edge_value[(u, v)] = domain_dti.get(n, np.nan)

    # Domain -> Fine（用 Fine/fine DTI）
    for n, fines in domain_map.items():
        n = _s(n)
        u = f"D|{n}"
        for f in fines:
            f = _s(f)
            v = f"F|{f}"
            edges.append((u, v))
            edge_value[(u, v)] = region_dti.get(f, np.nan)

    # -------- 5) 计算坐标 (x, y)：保证空 domain 也有位置 ----------
    # 组织成 coarse -> domain -> fine
    tree_struct = {c: {} for c in coarse_ids}
    for c in coarse_ids:
        normals = [_s(n) for n in coarse_map.get(c, []) if _s(n) in normal_ids]
        for n in sorted(normals, key=lambda x: int(x) if x.isdigit() else x):
            fines = [_s(f) for f in domain_map.get(n, []) if _s(f) in fine_ids]
            tree_struct[c][n] = sorted(fines, key=lambda x: int(x) if x.isdigit() else x)

    pos = {}
    fine_x_map = {}
    normal_placeholder_x = {}  # domain 没 fine 时，用它占位
    current_x = 0.0

    for c in coarse_ids:
        normals = list(tree_struct[c].keys())
        for n in normals:
            fines = tree_struct[c][n]
            if not fines:
                # 占位：让这个 domain 不会掉到 x=0
                normal_placeholder_x[n] = current_x
                current_x += 1.0
            else:
                for f in fines:
                    fine_x_map[f"F|{f}"] = current_x
                    current_x += 1.0 + gap_fine_within_domain
            current_x += gap_between_domains
        current_x += gap_between_coarse

    # Fine (y=-2)
    for nid, info in nodes.items():
        if info["level"] == 2:
            pos[nid] = (fine_x_map.get(nid, np.nan), -2)

    # Normal (y=-1)
    for n in normal_ids:
        n_id = f"D|{n}"
        children = [f"F|{_s(f)}" for f in domain_map.get(n, [])]
        child_xs = [pos[ch][0] for ch in children if (ch in pos and not np.isnan(pos[ch][0]))]
        if child_xs:
            pos[n_id] = (float(np.mean(child_xs)), -1)
        else:
            pos[n_id] = (float(normal_placeholder_x.get(n, 0.0)), -1)

    # Coarse (y=0)
    for c in coarse_ids:
        c_id = f"C|{c}"
        children = [f"D|{_s(n)}" for n in coarse_map.get(c, [])]
        child_xs = [pos[ch][0] for ch in children if (ch in pos and not np.isnan(pos[ch][0]))]
        pos[c_id] = (float(np.mean(child_xs)), 0) if child_xs else (0.0, 0)

    # -------- 6) x 归一化到 [0,1]（更紧凑美观） ----------
    xs = np.array([p[0] for p in pos.values() if not np.isnan(p[0])], dtype=float)
    if xs.size > 0 and xs.max() > xs.min():
        xmin, xmax = xs.min(), xs.max()
        for k in list(pos.keys()):
            x, y = pos[k]
            if np.isnan(x):
                continue
            pos[k] = ((x - xmin) / (xmax - xmin), y)
    else:
        for k in list(pos.keys()):
            x, y = pos[k]
            if not np.isnan(x):
                pos[k] = (0.5, y)

    # -------- 7) 颜色映射（inferno） ----------
    vals = np.array([v for v in edge_value.values() if not np.isnan(v)], dtype=float)
    if vals.size == 0:
        norm = Normalize(vmin=0, vmax=1)
    else:
        norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
    cmap = plt.get_cmap(cmap_name)

    # -------- 8) 绘图 ----------
    fig, ax = plt.subplots(figsize=figsize)

    def draw_sankey_ribbon(ax, p1, p2, color, alpha=0.5, width_px=8, n=30, zorder=1, outline=False, outline_lw=0.4):
        """
        在 ax 上画一条“桑基风格”的带状曲线（ribbon），连接 p1->p2。
        - width_px: 带宽（像素）
        - n: 采样点数，越大越平滑但越慢
        """
        x1, y1 = p1
        x2, y2 = p2
        midy = (y1 + y2) / 2.0
    
        # 中心线贝塞尔控制点（数据坐标）
        P0 = np.array([x1, y1], float)
        P1 = np.array([x1, midy], float)
        P2 = np.array([x2, midy], float)
        P3 = np.array([x2, y2], float)
    
        # 采样中心线点（数据坐标）
        t = np.linspace(0, 1, n)
        B = ((1-t)**3)[:, None]*P0 + (3*(1-t)**2*t)[:, None]*P1 + (3*(1-t)*t**2)[:, None]*P2 + (t**3)[:, None]*P3  # (n,2)
    
        # 转到显示坐标（像素），方便做“固定像素宽度”的偏移
        B_disp = ax.transData.transform(B)  # (n,2) in pixels
    
        # 用相邻点差分近似切向量（显示坐标）
        d = np.gradient(B_disp, axis=0)
        # 法向量 = (-dy, dx)
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
        nrm_len = np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        nrm = nrm / nrm_len
    
        # 上下边界（显示坐标）
        off = (width_px / 2.0) * nrm
        top_disp = B_disp + off
        bot_disp = B_disp - off
    
        # 变回数据坐标
        inv = ax.transData.inverted()
        top = inv.transform(top_disp)
        bot = inv.transform(bot_disp)
    
        # 构造闭合多边形：top 正向 + bot 反向
        poly = np.vstack([top, bot[::-1], top[0:1]])  # close
    
        # Path codes
        codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO]*(len(poly)-2) + [mpath.Path.CLOSEPOLY]
        path = mpath.Path(poly, codes)
    
        patch = mpatches.PathPatch(
            path,
            facecolor=color,
            edgecolor=(color if outline else "none"),
            lw=(outline_lw if outline else 0),
            alpha=alpha,
            zorder=zorder
        )
        ax.add_patch(patch)


    for u, v in edges:
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if np.isnan(x1) or np.isnan(x2):
            continue
        val = edge_value.get((u, v), np.nan)
        color = "#CFCFCF" if np.isnan(val) else cmap(norm(val))  
        val = edge_value.get((u, v), np.nan)


        # --- 1) 颜色（仍然由 val 决定） ---
        if np.isnan(val):
            color = "#CFCFCF"
            t = None
        else:
            color = cmap(norm(val))
            t = float(norm(val))  # 0~1

        # --- 2) ✅ 两个梯度的 wpx：按边类型区分 ---
        is_coarse_to_domain = u.startswith("C|") and v.startswith("D|")
        is_domain_to_fine   = u.startswith("D|") and v.startswith("F|")
    
        if np.isnan(val):
            # 缺失值：也按层级给两档默认宽度
            wpx = 12 if is_coarse_to_domain else 7
        else:
            if is_coarse_to_domain:
                # 粗：比如 [10, 22] px
                wpx = 5 + 12 * t
            elif is_domain_to_fine:
                # 细：比如 [4, 10] px
                wpx = 2 + 4 * t
            else:
                # 兜底（如果未来出现别的边类型）
                wpx = 6 + 8 * t

        draw_sankey_ribbon(
            ax,
            (x1, y1),
            (x2, y2),
            color=color,
            alpha=edge_alpha,
            width_px=wpx,
            n=25,          # 采样点数
            zorder=1,
            outline=True  # 需要边界线就 True
        )

    # 节点颜色（你原先的风格）
    colors = {0: "#2E8B57", 1: "#E47734", 2: "#6A5ACD"}

    # 画点 + 内部标签
    for nid, (x, y) in pos.items():
        level = nodes[nid]["level"]
        label = nodes[nid]["label"]
        color = colors.get(level, "black")

        if level == 0:
            size = node_size_base * 1.6
        elif level == 1:
            size = node_size_base * 1.25
        else:
            size = node_size_base * 1.0

        ax.scatter(x, y, s=size, c=color, edgecolor="black", linewidth=0.6, zorder=2)

        # 字体：两位数字放得下；更长就缩一点
        current_fs = fontsize if len(label) <= 2 else max(6, fontsize - 2)
        ax.text(x, y, label, ha="center", va="center",
                color="white", fontsize=current_fs, zorder=3)

    # y 轴层级标签
    ax.set_yticks([0, -1, -2])
    ax.set_yticklabels(["Coarse", "Domain", "fine"], fontsize=12)

    # 美化：去边框/去 x 轴
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.tick_params(left=False)

    # colorbar
    if show_colorbar and vals.size > 0:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("DTI", fontsize=11)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, 'Domain_Hierarchy_Tree_DTI.png')
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    return 


def _cast_group_to_dtype(adata, key, group_str):
    """
    把 '8' 转成 obs[key] categories 的 dtype（int 或 str），避免 sc.pl.spatial(groups=...) 匹配失败。
    """
    s = str(group_str)
    ser = adata.obs[key]

    # category：用 categories 的 dtype 判断
    if str(ser.dtype) == "category":
        cats = ser.cat.categories
        if np.issubdtype(cats.dtype, np.integer):
            return int(s)
        return s

    # 非 category：尽量按原 dtype
    dt = ser.dtype
    if np.issubdtype(dt, np.integer):
        return int(s)
    return s


def plot_domain_tree_spatial(
    adata,
    spot_size=100,
    width_per_node=1.2,
    height=10,
    inset_size=3,
    gene_tree=None,
    cmap_name='inferno',
    output_dir=None,
    palette=None,
    dpi=600,
    img_key=None
):
    """
    ✅ 节点顺序严格参考 plot_tree_inside_labels_with_dti：
      - coarse 按 coarse_ids 排序
      - 每个 coarse 内 domain 按排序
      - 每个 domain 内 fine 按排序并连续铺开 x
      - 每个 domain/ coarse 的 x 取子节点 x 的均值
      - gap_between_domains / gap_between_coarse 的插入位置与参考函数一致

    节点渲染：
      - 每个节点不是圆点，而是一个 sc.pl.spatial 的高亮 inset
    边：
      - Coarse->Domain 用 domain 的聚合 DTI 上色
      - Domain->Fine 用 fine 的聚合 DTI 上色
    """
    _adata=adata.copy()

    domain_map_key="domain_map"
    coarse_map_key="domain_coarse_map"
    fine_dti_col="fine_DTI"
    domain_dti_col="domain_DTI"
    fine_col="fine"

    fontsize=11

    # ✅ 顺序/布局参数：完全对齐 plot_tree_inside_labels_with_dti 的 step5
    gap_fine_within_domain=1.5
    gap_between_domains=0.8
    gap_between_coarse=1.5
    y_step=5.0

    # 视觉参数
    edge_alpha=0.9
    show_colorbar=True

    # ---------- 0) 检查 ----------
    if domain_map_key not in _adata.uns or coarse_map_key not in _adata.uns:
        raise KeyError(f"adata.uns 中未找到映射表: {domain_map_key} 或 {coarse_map_key}")
    if fine_col not in _adata.obs:
        raise KeyError(f"adata.obs 缺少列：{fine_col}")
    if fine_dti_col not in _adata.obs:
        raise KeyError(f"adata.obs 缺少列：{fine_dti_col}")
    if domain_dti_col not in _adata.obs:
        raise KeyError(f"adata.obs 缺少列：{domain_dti_col}")

    domain_map_raw = _adata.uns[domain_map_key]   # domain -> [fine]
    coarse_map_raw = _adata.uns[coarse_map_key]   # coarse -> [domain]

    def _s(x): return str(x)

    # ✅ 与参考函数一致：数字优先的排序
    def _sort_key(x):
        x = str(x)
        return int(x) if x.isdigit() else x

    # ---------- 1) 统一映射为 str-key/str-value（避免 1 vs "1"） ----------
    domain_map = { _s(d): [_s(f) for f in fs] for d, fs in domain_map_raw.items() }
    coarse_map = { _s(c): [_s(d) for d in ds] for c, ds in coarse_map_raw.items() }

    # fine->domain 映射（用于推断 spot 的 domain）
    fine_to_domain = {}
    for d, fs in domain_map.items():
        for f in fs:
            if f in fine_to_domain and fine_to_domain[f] != d:
                raise ValueError(f"fine {f} 同时属于 domain {fine_to_domain[f]} 和 {d}")
            fine_to_domain[f] = d

    # domain->coarse 映射（用于推断 spot 的 coarse）
    dom_to_coarse = {}
    for c, ds in coarse_map.items():
        for d in ds:
            if d in dom_to_coarse and dom_to_coarse[d] != c:
                raise ValueError(f"domain {d} 同时属于 coarse {dom_to_coarse[d]} 和 {c}")
            dom_to_coarse[d] = c

    # ---------- 2) 临时列：domain/coarse（用于 inset 的 sc.pl.spatial 高亮） ----------
    d_col, c_col = "tmp_tree_domain", "tmp_tree_coarse"
    region_series = _adata.obs[fine_col].astype(str)
    _adata.obs[d_col] = region_series.map(fine_to_domain).astype(str)
    _adata.obs[c_col] = _adata.obs[d_col].map(dom_to_coarse).astype(str)

    # ---------- 3) 计算每个 domain / fine 的聚合 DTI（对齐参考函数） ----------
    region_dti = _adata.obs.groupby(region_series)[fine_dti_col].mean()
    domain_dti = _adata.obs.groupby(_adata.obs[d_col].astype(str))[domain_dti_col].mean()
    region_dti = region_dti.to_dict()  # key=str(fine)
    domain_dti = domain_dti.to_dict()  # key=str(domain)

    # ---------- 4) 准备节点集合（coarse/domain/fine ids） ----------
    coarse_ids = sorted(list(coarse_map.keys()), key=_sort_key)
    normal_ids = sorted({d for ds in coarse_map.values() for d in ds}, key=_sort_key)
    fine_ids   = sorted({f for fs in domain_map.values() for f in fs}, key=_sort_key)

    # ---------- 5) ✅ 关键：坐标生成完全照搬 plot_tree_inside_labels_with_dti ----------
    # 组织成 coarse -> domain -> fine（并排序）
    tree_struct = {c: {} for c in coarse_ids}
    for c in coarse_ids:
        normals = [n for n in coarse_map.get(c, []) if n in normal_ids]
        for n in sorted(normals, key=_sort_key):
            fines = [f for f in domain_map.get(n, []) if f in fine_ids]
            tree_struct[c][n] = sorted(fines, key=_sort_key)

    pos = {}
    fine_x_map = {}
    normal_placeholder_x = {}
    current_x = 0.0

    for c in coarse_ids:
        normals = list(tree_struct[c].keys())  # 已按插入顺序（排序后）写入
        for n in normals:
            fines = tree_struct[c][n]
            if not fines:
                # domain 没 fine：占位
                normal_placeholder_x[n] = current_x
                current_x += 1.0
            else:
                for f in fines:
                    fine_x_map[f"F|{f}"] = current_x
                    current_x += 1.0 + gap_fine_within_domain
            current_x += gap_between_domains
        current_x += gap_between_coarse

    # Fine (y = -2*y_step)
    for f in fine_ids:
        nid = f"F|{f}"
        pos[nid] = (fine_x_map.get(nid, np.nan), -2 * y_step)

    # Domain (y = -1*y_step)
    for n in normal_ids:
        n_id = f"D|{n}"
        children = [f"F|{f}" for f in domain_map.get(n, [])]
        child_xs = [pos[ch][0] for ch in children if (ch in pos and not np.isnan(pos[ch][0]))]
        if child_xs:
            pos[n_id] = (float(np.mean(child_xs)), -1 * y_step)
        else:
            pos[n_id] = (float(normal_placeholder_x.get(n, 0.0)), -1 * y_step)

    # Coarse (y = 0)
    for c in coarse_ids:
        c_id = f"C|{c}"
        children = [f"D|{n}" for n in coarse_map.get(c, [])]
        child_xs = [pos[ch][0] for ch in children if (ch in pos and not np.isnan(pos[ch][0]))]
        pos[c_id] = (float(np.mean(child_xs)), 0.0) if child_xs else (0.0, 0.0)

    # x 归一化到 [0,1]（对齐参考函数）
    xs = np.array([p[0] for p in pos.values() if not np.isnan(p[0])], dtype=float)
    if xs.size > 0 and xs.max() > xs.min():
        xmin, xmax = xs.min(), xs.max()
        for k in list(pos.keys()):
            x, y = pos[k]
            if np.isnan(x):
                continue
            pos[k] = ((x - xmin) / (xmax - xmin), y)
    else:
        for k in list(pos.keys()):
            x, y = pos[k]
            if not np.isnan(x):
                pos[k] = (0.5, y)

    # ---------- 6) 构建边 + 记录边的 DTI（对齐参考函数：边颜色=子节点DTI） ----------
    edges = []
    edge_value = {}

    # Coarse -> Domain（用 Domain DTI）
    for c, normals in coarse_map.items():
        u = f"C|{c}"
        for n in normals:
            v = f"D|{n}"
            edges.append((u, v))
            edge_value[(u, v)] = domain_dti.get(n, np.nan)

    # Domain -> Fine（用 Fine/fine DTI）
    for n, fines in domain_map.items():
        u = f"D|{n}"
        for f in fines:
            v = f"F|{f}"
            edges.append((u, v))
            edge_value[(u, v)] = region_dti.get(f, np.nan)

    # ---------- 7) 颜色映射 ----------
    vals = np.array([v for v in edge_value.values() if not np.isnan(v)], dtype=float)
    if vals.size == 0:
        norm = Normalize(vmin=0, vmax=1)
    else:
        norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
    cmap = plt.get_cmap(cmap_name)

    # # ---------- 8) 绘图：自适应画图----------
    total_x_units = 0.0
    for c in coarse_ids:
        for n in sorted(coarse_map.get(c, []), key=_sort_key):
            fines = sorted(domain_map.get(n, []), key=_sort_key)
            if not fines:
                total_x_units += 1.0 # 占位符
            else:
                # 每个 fine 节点占 1.0，加上 fine 之间的 gap
                total_x_units += len(fines)
            total_x_units += gap_between_domains
        total_x_units += gap_between_coarse
    dynamic_width = max(10, total_x_units * width_per_node) 
    figsize = (dynamic_width, height)
    print(figsize)
    fig, ax = plt.subplots(figsize=figsize)


    def draw_sankey_ribbon(ax, p1, p2, color, alpha=0.5, width_px=8, n=30, zorder=1, outline=False, outline_lw=0.4):
        """
        在 ax 上画一条“桑基风格”的带状曲线（ribbon），连接 p1->p2。
        - width_px: 带宽（像素）
        - n: 采样点数，越大越平滑但越慢
        """
        x1, y1 = p1
        x2, y2 = p2
        midy = (y1 + y2) / 2.0
    
        # 中心线贝塞尔控制点（数据坐标）
        P0 = np.array([x1, y1], float)
        P1 = np.array([x1, midy], float)
        P2 = np.array([x2, midy], float)
        P3 = np.array([x2, y2], float)
    
        # 采样中心线点（数据坐标）
        t = np.linspace(0, 1, n)
        B = ((1-t)**3)[:, None]*P0 + (3*(1-t)**2*t)[:, None]*P1 + (3*(1-t)*t**2)[:, None]*P2 + (t**3)[:, None]*P3  # (n,2)
    
        # 转到显示坐标（像素），方便做“固定像素宽度”的偏移
        B_disp = ax.transData.transform(B)  # (n,2) in pixels
    
        # 用相邻点差分近似切向量（显示坐标）
        d = np.gradient(B_disp, axis=0)
        # 法向量 = (-dy, dx)
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
        nrm_len = np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        nrm = nrm / nrm_len
    
        # 上下边界（显示坐标）
        off = (width_px / 2.0) * nrm
        top_disp = B_disp + off
        bot_disp = B_disp - off
    
        # 变回数据坐标
        inv = ax.transData.inverted()
        top = inv.transform(top_disp)
        bot = inv.transform(bot_disp)
    
        # 构造闭合多边形：top 正向 + bot 反向
        poly = np.vstack([top, bot[::-1], top[0:1]])  # close
    
        # Path codes
        codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO]*(len(poly)-2) + [mpath.Path.CLOSEPOLY]
        path = mpath.Path(poly, codes)
    
        patch = mpatches.PathPatch(
            path,
            facecolor=color,
            edgecolor=(color if outline else "none"),
            lw=(outline_lw if outline else 0),
            alpha=alpha,
            zorder=zorder
        )
        ax.add_patch(patch)


    for u, v in edges:
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if np.isnan(x1) or np.isnan(x2):
            continue
        val = edge_value.get((u, v), np.nan)
        color = "#CFCFCF" if np.isnan(val) else cmap(norm(val))  
        val = edge_value.get((u, v), np.nan)
        # if np.isnan(val):
        #     color = "#CFCFCF"
        #     wpx = 6  # 缺失值用较细
        # else:
        #     color = cmap(norm(val))
        #     # ✅ 桑基味：让宽度随 val 变化（你也可以换成 spot 数等）
        #     t = float(norm(val))  # 0~1
        #     wpx = 4 + 10 * t       # [4,14] px，可调

        # --- 1) 颜色（仍然由 val 决定） ---
        if np.isnan(val):
            color = "#CFCFCF"
            t = None
        else:
            color = cmap(norm(val))
            t = float(norm(val))  # 0~1

        # --- 2) ✅ 两个梯度的 wpx：按边类型区分 ---
        is_coarse_to_domain = u.startswith("C|") and v.startswith("D|")
        is_domain_to_fine   = u.startswith("D|") and v.startswith("F|")
    
        if np.isnan(val):
            # 缺失值：也按层级给两档默认宽度
            wpx = 12 if is_coarse_to_domain else 7
        else:
            if is_coarse_to_domain:
                # 粗：比如 [10, 22] px
                wpx = 10 + 12 * t
            elif is_domain_to_fine:
                # 细：比如 [4, 10] px
                wpx = 4 + 4 * t
            else:
                # 兜底（如果未来出现别的边类型）
                wpx = 6 + 8 * t

        draw_sankey_ribbon(
            ax,
            (x1, y1),
            (x2, y2),
            color=color,
            alpha=edge_alpha,
            width_px=wpx,
            n=25,          # 采样点数
            zorder=1,
            outline=True  # 需要边界线就 True
        )


    colors = {0: "#2E8B57", 1: "#E47734", 2: "#6A5ACD"}
    
    # ---------- 9) 节点改成 inset spatial 图 ----------
    # for nid, (nx, ny) in pos.items():
    #     if nid.startswith("C|"):
    #         lvl = 0
    #         col = c_col
    #         label = nid.split("|", 1)[1]
    #         edge_id = f"coarse:{label}"
    #     elif nid.startswith("D|"):
    #         lvl = 1
    #         col = d_col
    #         label = nid.split("|", 1)[1]
    #         edge_id = f"domain:{label}"
    #     else:
    #         lvl = 2
    #         col = fine_col
    #         label = nid.split("|", 1)[1]
    #         edge_id = f"fine:{label}"

    for nid, (nx, ny) in pos.items():
        if nid.startswith("C|"):
            lvl = 0
            col = "domain_coarse"   # ✅ 直接指向您预设好颜色的真实列
            label = nid.split("|", 1)[1]
            edge_id = f"coarse:{label}"
        elif nid.startswith("D|"):
            lvl = 1
            col = "domain"          # ✅ 直接指向您预设好颜色的真实列
            label = nid.split("|", 1)[1]
            edge_id = f"domain:{label}"
        else:
            lvl = 2
            col = fine_col          # fine 本身就是原列，保持不变
            label = nid.split("|", 1)[1]
            edge_id = f"fine:{label}"
            
        sub_ax = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=(nx - inset_size / 2, ny - inset_size / 2, inset_size, inset_size),
            bbox_transform=ax.transData,
            borderpad=0
        )

        # c) 计算该节点的权重评分 (调用之前重构的计算函数)
        # 注意：这里加上 try-except 预防某些单支节点没有计算过 score
         # 假设 calculate_edge_scores 已在外部定义
        if gene_tree is not None:
            score_values = TGL_module.calculate_edge_scores(
                _adata, 
                gene_tree, 
                edge_id, 
                Knee_S=3 # 使用稍微严格的肘点
            )
            score_name = f"tmp_score_{edge_id.replace(':', '_')}"
            _adata.obs[score_name] = score_values
            
            # 绘图：展示 Score 梯度
            sc.pl.spatial(
                _adata,
                color=score_name,
                img_key=img_key,
                spot_size=spot_size,
                frameon=False,
                ax=sub_ax,
                show=False,
                title='',
                cmap="RdYlBu_r", 
                legend_loc=None,
                colorbar_loc=None,
            )
        else:

            gid = _cast_group_to_dtype(_adata, col, label)

            # 1) inset 空间图本身：不要 title
            sc.pl.spatial(
                _adata,
                color=col,
                img_key=img_key,
                groups=[gid],
                spot_size=spot_size,
                frameon=False,
                ax=sub_ax,
                show=False,
                title='',
                legend_loc=None,
                colorbar_loc=None,
            )
            

        # 2) 清理子图
        sub_ax.set_xticks([]); sub_ax.set_yticks([])
        sub_ax.set_xlabel(""); sub_ax.set_ylabel("")
        sub_ax.set_title("")   

        for spine in sub_ax.spines.values():
            spine.set_edgecolor(colors.get(lvl, "black"))
            spine.set_linewidth(2)
    
        s_label = str(label)
        current_fs = fontsize if len(s_label) <= 2 else max(6, fontsize - 2)
    
        ax.text(
            nx, ny+1.8, s_label,
            ha="center", va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
            zorder=10,
            # 关键：圆形底色（看起来就是“圈中数字”）
            bbox=dict(
                boxstyle="circle,pad=0.28",
                fc=colors.get(lvl, "black"),
                ec="black",
                lw=0.6,
                alpha=0.95
            )
        )

    # ---------- 10) 全局修饰 ----------
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-2 * y_step - inset_size, 0 + inset_size)

    ax.set_yticks([0, -1 * y_step, -2 * y_step])
    ax.set_yticklabels(["Coarse", "Domain", "Fine"], fontsize=14, fontweight="bold")

    for s in ax.spines.values():
        s.set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.tick_params(left=False)

    if show_colorbar and vals.size > 0:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.02)
        cbar.set_label("DTI Intensity", fontsize=12)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if gene_tree is not None:
            plt.savefig(os.path.join(output_dir, "Gene_tree_spatial.png"), dpi=dpi, bbox_inches="tight")
        else:
            plt.savefig(os.path.join(output_dir, "Domain_tree_spatial.png"), dpi=dpi, bbox_inches="tight")

    # # 清理临时列
    # _adata.obs.drop(columns=[d_col, c_col], inplace=True, errors="ignore")

    plt.show()
    return fig, ax

############################################################################################

def plot_upper_bound_tracking_multi(
    csv_path,
    output_dir,
    max_steps_list=(600, 100),
    seed_range=(1, 200),
    min_points=10,
    smooth_factor=1,
    figsize=(14, 6),
    dpi=1000,
):
    """同一份 log csv，用不同 max_steps 画多张 upper bound tracking。"""
    for max_steps in max_steps_list:
        plot_upper_bound_tracking(
            csv_path,
            output_dir=output_dir,
            seed_range=seed_range,
            min_points=min_points,
            max_steps=max_steps,
            smooth_factor=smooth_factor,
            figsize=figsize,
            dpi=dpi,
        )


def plot_stage1_fig(
    adata,
    output_dir,
    spot_size,
    bins=50,
    show_median=False,
    # gradient field
    grad_field_step=1,
    grad_field_scale=150,
    grad_field_flip_y=True,
    grad_field_show_points=True,
    grad_field_point_alpha=0.3,
    grad_field_point_size=10,
    grad_field_figsize=(5, 5),
    grad_field_dpi=1000,
    # gradient magnitude
    grad_mag_flip_y=True,
    grad_mag_point_size=5,
    grad_mag_figsize=(5, 5),
    grad_mag_dpi=600,
    grad_mag_cmap="viridis",
    # upper bound tracking
    seed_range=(1, 200),
    ub_min_points=10,
    ub_max_steps_list=(600, 100),
    ub_smooth_factor=1,
    ub_figsize=(14, 6),
    ub_dpi=1000,
    # spatial dpi
    spatial_dpi=600,
    show=False,
):
    """
    Stage1 产物图统一封装（你贴的那一组图）。
    - 默认保存到 output_dir（scanpy spatial 这几张我这里也保存为 png）
    - stEDGE.plot 的函数本身通常会保存到 output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) p_edge / a_p_edge spatial（如果字段存在才画）
    colors = [c for c in ["p_edge", "a_p_edge"] if c in adata.obs]
    if len(colors) > 0:
        plot_spatial(
            adata,
            colors=colors,
            titles=colors,
            spot_size=spot_size,
            cmap="inferno",
            figsize=(5, 5),
            colorbar_loc="right",
            show=show,
            save_path=os.path.join(output_dir, "spatial_p_edge.png"),
            dpi=spatial_dpi,
        )

    # 2) p_edge 分布图（stEDGE.plot，可复用）
    if "p_edge" in adata.obs:
        plot_p_edge_distribution(
            adata,
            output_dir=output_dir,
            bins=bins,
            color="skyblue",
            show_median=show_median,
        )

    # 3) labeled_seeds spatial
    if "labeled_seeds" in adata.obs:
        plot_spatial(
            adata,
            colors=["labeled_seeds"],
            titles=["labeled_seeds"],
            spot_size=spot_size,
            cmap="inferno",
            figsize=(10, 10),
            colorbar_loc="right",
            show=show,
            save_path=os.path.join(output_dir, "spatial_labeled_seeds.png"),
            dpi=spatial_dpi,
        )

    # 4) region_label spatial
    if "region_label" in adata.obs:
        plot_spatial(
            adata,
            colors=["region_label"],
            titles=["region_label"],
            spot_size=spot_size,
            figsize=(5, 5),
            colorbar_loc=None,
            legend_loc="right margin",
            show=show,
            save_path=os.path.join(output_dir, "spatial_region_label.png"),
            dpi=spatial_dpi,
        )

    # 5) gradient field / magnitude（stEDGE.plot，可复用）
    plot_gradient_field(
        adata,
        output_dir=output_dir,
        step=grad_field_step,
        scale=grad_field_scale,
        flip_y=grad_field_flip_y,
        show_points=grad_field_show_points,
        point_alpha=grad_field_point_alpha,
        point_size=grad_field_point_size,
        figsize=grad_field_figsize,
        dpi=grad_field_dpi,
    )

    plot_gradient_magnitude(
        adata,
        output_dir=output_dir,
        flip_y=grad_mag_flip_y,
        point_size=grad_mag_point_size,
        figsize=grad_mag_figsize,
        dpi=grad_mag_dpi,
        cmap=grad_mag_cmap,
    )

    # 6) upper bound tracking（stEDGE.plot，可复用）
    csv_path = os.path.join(output_dir, "region_growing_log.csv")
    if os.path.exists(csv_path):
        plot_upper_bound_tracking_multi(
            csv_path,
            output_dir=output_dir,
            max_steps_list=ub_max_steps_list,
            seed_range=seed_range,
            min_points=ub_min_points,
            smooth_factor=ub_smooth_factor,
            figsize=ub_figsize,
            dpi=ub_dpi,
        )

    # 7) con_region / fine spatial
    if "con_region" in adata.obs:
        plot_spatial(
            adata,
            colors=["con_region"],
            titles=["con_region"],
            spot_size=spot_size,
            figsize=(5, 5),
            colorbar_loc=None,
            legend_loc="right margin",
            show=show,
            save_path=os.path.join(output_dir, "spatial_con_region.png"),
            dpi=spatial_dpi,
        )

    if "fine" in adata.obs:
        plot_spatial(
            adata,
            colors=["fine"],
            titles=["fine"],
            spot_size=spot_size,
            figsize=(5, 5),
            colorbar_loc=None,
            legend_loc="right margin",
            show=show,
            save_path=os.path.join(output_dir, "spatial_fine.png"),
            dpi=spatial_dpi,
        )


def plot_stage2_fig(
    adata,
    output_dir,
    spot_size,
    # keys
    fine_key="fine",
    domain_key="domain",
    coarse_key="domain_coarse",
    # graph params
    min_sim_threshold=0.1,
    graph_figsize=(8, 6),
    boundary_graph_figsize=(8, 6),
    boundary_node_size=300,
    boundary_title="Domain boundary graph",
    # gain log params
    gain_figsize=(6, 3),
    # tree params
    tree_figsize=(14, 4),
    tree_figsize_spatial=(25, 10),
    inset_size=3,
    # spatial dpi/show
    spatial_dpi=600,
    show=False,
    verbose=True,
):
    """
    Stage2 产物图统一封装（你贴的那一组图）。
    - spatial 图：使用 plot_spatial 保存为 png
    - graph / gain log / tree：复用本模块已有 plot 函数（通常内部会保存到 output_dir）
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 1) fine_DTI / fine_TS spatial ----------
    fine_dti_col = f"{fine_key}_DTI"
    fine_ts_col = f"{fine_key}_TS"
    colors = [c for c in [fine_dti_col, fine_ts_col] if c in adata.obs]
    if len(colors) > 0:
        plot_spatial(
            adata,
            colors=colors,
            titles=colors,
            spot_size=spot_size,
            cmap="inferno",
            figsize=(5, 5),
            show=show,
            save_path=os.path.join(output_dir, f"spatial_{fine_key}_DTI_TS.png"),
            dpi=spatial_dpi,
        )
    elif verbose:
        print(f"[plot_stage2_fig] Skip spatial {fine_dti_col}/{fine_ts_col}: columns not found.")

    # ---------- 2) domain graph for fine ----------
    try:
        plot_domain_graph(
            adata,
            domain_key=fine_key,
            min_sim_threshold=min_sim_threshold,
            figsize=graph_figsize,
            output_dir=output_dir,
        )
    except Exception as e:
        if verbose:
            print(f"[plot_stage2_fig] Skip plot_domain_graph({fine_key}): {e}")

    # ---------- 3) domain boundary graph ----------
    try:
        plot_domain_boundary_graph(
            adata,
            figsize=boundary_graph_figsize,
            node_size=boundary_node_size,
            title=boundary_title,
            output_dir=output_dir,
        )
    except Exception as e:
        if verbose:
            print(f"[plot_stage2_fig] Skip plot_domain_boundary_graph: {e}")

    # ---------- 4) domain spatial ----------
    if domain_key in adata.obs:
        plot_spatial(
            adata,
            colors=[domain_key],
            titles=[domain_key],
            spot_size=spot_size,
            figsize=(5, 5),
            legend_loc="right margin",
            show=show,
            save_path=os.path.join(output_dir, f"spatial_{domain_key}.png"),
            dpi=spatial_dpi,
        )
    elif verbose:
        print(f"[plot_stage2_fig] Skip spatial {domain_key}: column not found.")

    # ---------- 5) domain_DTI / domain_TS spatial ----------
    dom_dti_col = f"{domain_key}_DTI"
    dom_ts_col = f"{domain_key}_TS"
    colors = [c for c in [dom_dti_col, dom_ts_col] if c in adata.obs]
    if len(colors) > 0:
        plot_spatial(
            adata,
            colors=colors,
            titles=colors,
            spot_size=spot_size,
            cmap="inferno",
            figsize=(5, 5),
            show=show,
            save_path=os.path.join(output_dir, f"spatial_{domain_key}_DTI_TS.png"),
            dpi=spatial_dpi,
        )
    elif verbose:
        print(f"[plot_stage2_fig] Skip spatial {dom_dti_col}/{dom_ts_col}: columns not found.")

    # ---------- 6) domain graph for domain (two modes) ----------
    try:
        plot_domain_graph(
            adata,
            domain_key=domain_key,
            color_mode="dti",
            min_sim_threshold=min_sim_threshold,
            figsize=graph_figsize,
            output_dir=output_dir,
        )
    except Exception as e:
        if verbose:
            print(f"[plot_stage2_fig] Skip plot_domain_graph({domain_key}): {e}")

    try:
        plot_domain_graph(
            adata,
            domain_key=domain_key,
            color_mode="cluster",
            min_sim_threshold=min_sim_threshold,
            figsize=graph_figsize,
            legend=True,
            output_dir=output_dir,
        )
    except Exception as e:
        if verbose:
            print(f"[plot_stage2_fig] Skip plot_domain_graph({domain_key}, color_mode='cluster'): {e}")

    # ---------- 7) coarse domain spatial ----------
    if coarse_key in adata.obs:
        plot_spatial(
            adata,
            colors=[coarse_key],
            titles=[coarse_key],
            spot_size=spot_size,
            legend_loc="right margin",
            figsize=(5, 5),
            show=show,
            save_path=os.path.join(output_dir, f"spatial_{coarse_key}.png"),
            dpi=spatial_dpi,
        )
    elif verbose:
        print(f"[plot_stage2_fig] Skip spatial {coarse_key}: column not found.")

    # ---------- 8) domain gain log ----------
    if f"{domain_key}_gain_log" in adata.uns:
        try:
            plot_domain_gain_log(
                adata,
                uns_key=f"{domain_key}_gain_log",
                gain_key="best_gain",
                title="Best Gain per Domain Merge Step",
                ylabel="Best Gain",
                figsize=gain_figsize,
                output_dir=output_dir,
            )
            plot_domain_gain_log(
                adata,
                uns_key=f"{domain_key}_gain_log",
                gain_key="entropy_after",
                title="Entropy per Domain Merge Step",
                ylabel="Entropy",
                figsize=gain_figsize,
                output_dir=output_dir,
            )
        except Exception as e:
            if verbose:
                print(f"[plot_stage2_fig] Skip plot_domain_gain_log: {e}")
    elif verbose:
        print(f"[plot_stage2_fig] Skip plot_domain_gain_log: adata.uns[{domain_key}_gain_log] not found.")

    # ---------- 9) tree plot inside labels with DTI ----------
    # 需要这些列存在才画（至少 DTI 列）
    required_cols = [fine_dti_col, dom_dti_col, fine_key]
    missing = [c for c in required_cols if c not in adata.obs]
    if len(missing) == 0:
        try:
            plot_domain_tree(
                adata,
                figsize=tree_figsize,
                output_dir=output_dir,
            )

            plot_domain_tree_spatial(
                adata,
                spot_size=spot_size,
                figsize=tree_figsize_spatial,
                inset_size=inset_size,
                output_dir=output_dir,
            )
        except Exception as e:
            if verbose:
                print(f"[plot_stage2_fig] Skip plot_tree_inside_labels_with_dti: {e}")
    elif verbose:
        print(f"[plot_stage2_fig] Skip plot_tree_inside_labels_with_dti: missing {missing}.")


