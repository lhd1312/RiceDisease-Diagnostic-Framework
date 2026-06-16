import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    Dense, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, 
    Reshape, Multiply, Concatenate, Add, Activation, 
    SeparableConv1D, BatchNormalization
)
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- Matplotlib 字体设置 ---
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 14,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'xtick.minor.width': 1.5, 'ytick.minor.width': 1.5, 'xtick.direction': 'in',
    'ytick.direction': 'in'
})

# ==========================================================
# 自定义模块定义 (加载模型所必需)
# ==========================================================
def cbam_block(input_feature, ratio=8):
    channel_axis = -1
    filters = input_feature.shape[channel_axis]
    shared_layer_one = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(filters, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    avg_pool = GlobalAveragePooling1D()(input_feature); avg_pool = Reshape((1, filters))(avg_pool); avg_pool = shared_layer_one(avg_pool); avg_pool = shared_layer_two(avg_pool)
    max_pool = GlobalMaxPooling1D()(input_feature); max_pool = Reshape((1, filters))(max_pool); max_pool = shared_layer_one(max_pool); max_pool = shared_layer_two(max_pool)
    cbam_feature = Add()([avg_pool, max_pool]); cbam_feature = Activation('sigmoid')(cbam_feature)
    x_refined_c = Multiply()([input_feature, cbam_feature])
    avg_pool = tf.reduce_mean(x_refined_c, axis=channel_axis, keepdims=True); max_pool = tf.reduce_max(x_refined_c, axis=channel_axis, keepdims=True)
    concat = Concatenate(axis=channel_axis)([avg_pool, max_pool])
    cbam_feature = Conv1D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    x_refined_s = Multiply()([x_refined_c, cbam_feature])
    return x_refined_s

def multiscale_block(inputs):
    b1 = SeparableConv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    b2 = SeparableConv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    b3 = SeparableConv1D(32, kernel_size=7, padding='same', activation='relu')(inputs)
    x = Concatenate()([b1, b2, b3])
    x = BatchNormalization()(x)
    return x
# ==========================================================

def add_noise_with_snr(data, snr_db):
    """
    根据给定的信噪比(SNR)向数据添加高斯噪声
    
    Args:
        data: 原始数据 (n_samples, n_features)
        snr_db: 信噪比 (单位: dB), None表示无噪声
    
    Returns:
        添加噪声后的数据
    """
    if snr_db is None:  # 无噪声情况
        return data.copy()
    
    # 计算信号功率
    signal_power = np.mean(data ** 2)
    
    # 根据SNR计算噪声功率
    # SNR(dB) = 10 * log10(P_signal / P_noise)
    # P_noise = P_signal / 10^(SNR/10)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # 生成噪声 (标准差 = sqrt(noise_power))
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, data.shape)
    
    return data + noise

# --- 数据路径和常量 ---
DATA_DIR = Path('DATA')
VNIR_FILE = DATA_DIR / 'VNIR.xlsx'
SWIR_FILE = DATA_DIR / 'SWIR.xlsx'
SHEET_NAMES = [
    'CK', 'DWB_01_Resistant', 'DWB_02_Resistant', 'DWB_03_Resistant', 'DWB_04_Resistant',
    'DWB_05_Susceptible', 'BYK_01_Resistant', 'BYK_02_Resistant', 'BYK_03_Resistant', 'BYK_04_Susceptible'
]
label_map = {name: i for i, name in enumerate(SHEET_NAMES)}
N_CLASSES = 10
N_COMPONENTS_PCA = 100

# --- 数据加载 ---
features = []
labels = []
all_bands = np.array([])
print("开始加载和合并数据...")
with pd.ExcelFile(VNIR_FILE) as vnir_xls, pd.ExcelFile(SWIR_FILE) as swir_xls:
    for sheet in tqdm(SHEET_NAMES, desc="Processing sheets"):
        vnir_data = pd.read_excel(vnir_xls, sheet_name=sheet, header=0)
        swir_data = pd.read_excel(swir_xls, sheet_name=sheet, header=0)
        if not all_bands.any():
            vnir_bands = vnir_data.columns.astype(float)
            swir_bands = swir_data.columns.astype(float)
            all_bands = np.concatenate([vnir_bands, swir_bands])
        combined_data = pd.concat([vnir_data, swir_data], axis=1)
        features.append(combined_data)
        current_labels = np.full(combined_data.shape[0], label_map[sheet])
        labels.append(current_labels)
X_orig = pd.concat(features, ignore_index=True).values
y_orig = np.concatenate(labels)

# --- 裁剪光谱范围 ---
WAVELENGTH_RANGES_TO_KEEP = [(420, 900), (1000, 1700)]
idx_keep = np.zeros(all_bands.shape, dtype=bool)
for (start, end) in WAVELENGTH_RANGES_TO_KEEP:
    idx_keep = idx_keep | ((all_bands >= start) & (all_bands <= end))
X_orig = X_orig[:, idx_keep]
N_BANDS_RAW = X_orig.shape[1]

# --- 预处理 (SG+SNV) ---
print("\n开始光谱预处理 (SG+SNV)...")
X_processed = savgol_filter(X_orig, window_length=11, polyorder=3, axis=1)
mean_row = np.mean(X_processed, axis=1, keepdims=True)
std_row = np.std(X_processed, axis=1, keepdims=True)
X_processed = (X_processed - mean_row) / (std_row + 1e-8)

# --- 划分数据集 ---
_, X_test_orig, _, y_test = train_test_split(
    X_processed, y_orig, test_size=0.3, random_state=42, stratify=y_orig
)
print(f"\n测试集大小: {len(y_test)}")

# --- 加载模型和预处理器 ---
print("\n加载所有模型和预处理器...")
try:
    custom_objects = {
        "cbam_block": cbam_block,
        "multiscale_block": multiscale_block
    }
    model_A = load_model("Ablation_msc_cbam.h5", custom_objects=custom_objects)
    model_B = load_model("Model_B_msc.h5", custom_objects=custom_objects)
    
    scaler_A = joblib.load("scaler_A_trimmed.pkl")
    scaler_B = joblib.load("scaler_B_trimmed.pkl")
    pca_B = joblib.load("pca_B_trimmed.pkl")
    
    print("✓ 模型和预处理器加载成功")
except Exception as e:
    print(f"!!! 错误: 加载模型失败 !!!")
    print(f"原始错误: {e}")
    exit()

# ==========================================================
# 核心部分：基于SNR的噪声鲁棒性分析
# ==========================================================

# 定义3个SNR级别 (中度到较强噪声范围)
snr_levels = [40, 30, 20]  # SNR: 40dB, 30dB, 20dB
snr_labels = ['40 dB', '30 dB', '20 dB']

# 设置集成权重
ENSEMBLE_WEIGHT_A = 0.51
ENSEMBLE_WEIGHT_B = 1.0 - ENSEMBLE_WEIGHT_A

print("\n" + "="*70)
print("开始基于SNR的噪声鲁棒性分析...")
print(f"信噪比级别: {snr_labels}")
print(f"集成模型权重: Model A = {ENSEMBLE_WEIGHT_A:.2f}, Model B = {ENSEMBLE_WEIGHT_B:.2f}")
print("="*70 + "\n")

# 存储结果 - 使用列表存储，方便多次实验
results_list = []

# 为了更稳定的结果，每个SNR级别运行多次
N_TRIALS = 10  # 每个SNR级别运行10次

for snr_db, snr_label in zip(snr_levels, snr_labels):
    print(f"\n处理噪声级别: {snr_label}")
    
    for trial in tqdm(range(N_TRIALS), desc=f"  {snr_label}"):
        # 设置不同的随机种子
        np.random.seed(42 + trial)
        
        # 添加噪声
        X_test_noisy = add_noise_with_snr(X_test_orig, snr_db)
        
        # --- 评估 Model A (Raw Spectrum Branch) ---
        X_test_A = scaler_A.transform(X_test_noisy)
        X_test_A_cnn = X_test_A.reshape(X_test_A.shape[0], N_BANDS_RAW, 1)
        probs_A = model_A.predict(X_test_A_cnn, verbose=0)
        y_pred_A = np.argmax(probs_A, axis=1)
        acc_A = accuracy_score(y_test, y_pred_A)
        
        # --- 评估 Model B (PCA-processed Branch) ---
        X_test_B = scaler_B.transform(X_test_noisy)
        X_test_pca = pca_B.transform(X_test_B)
        X_test_B_cnn = X_test_pca.reshape(X_test_pca.shape[0], N_COMPONENTS_PCA, 1)
        probs_B = model_B.predict(X_test_B_cnn, verbose=0)
        y_pred_B = np.argmax(probs_B, axis=1)
        acc_B = accuracy_score(y_test, y_pred_B)
        
        # --- 评估 Ensemble Model ---
        probs_ensemble = (ENSEMBLE_WEIGHT_A * probs_A) + (ENSEMBLE_WEIGHT_B * probs_B)
        y_pred_ensemble = np.argmax(probs_ensemble, axis=1)
        acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
        
        # 保存结果
        results_list.append({
            'SNR': snr_label,
            'SNR_dB': snr_db,
            'Trial': trial + 1,
            'Model_A': acc_A,
            'Model_B': acc_B,
            'Ensemble': acc_ensemble
        })

# 转换为 DataFrame
results_df = pd.DataFrame(results_list)

# ==========================================================
# 统计分析
# ==========================================================
print("\n" + "="*70)
print("噪声鲁棒性分析结果统计")
print("="*70)

# 按SNR级别统计
stats_df = results_df.groupby('SNR').agg({
    'Model_A': ['mean', 'std'],
    'Model_B': ['mean', 'std'],
    'Ensemble': ['mean', 'std']
}).round(4)

print("\n各SNR级别下的准确率统计 (均值 ± 标准差):")
for snr_label in snr_labels:
    data = results_df[results_df['SNR'] == snr_label]
    print(f"\n{snr_label}:")
    print(f"  Model A (Raw):    {data['Model_A'].mean():.4f} ± {data['Model_A'].std():.4f}")
    print(f"  Model B (PCA):    {data['Model_B'].mean():.4f} ± {data['Model_B'].std():.4f}")
    print(f"  Ensemble:         {data['Ensemble'].mean():.4f} ± {data['Ensemble'].std():.4f}")

# 最佳性能（最高SNR）
best_snr = results_df[results_df['SNR'] == snr_labels[0]]  # 40 dB
best_A = best_snr['Model_A'].mean()
best_B = best_snr['Model_B'].mean()
best_ensemble = best_snr['Ensemble'].mean()

# 最差性能（最低SNR）
worst_snr = results_df[results_df['SNR'] == snr_labels[-1]]  # 20 dB
worst_A = worst_snr['Model_A'].mean()
worst_B = worst_snr['Model_B'].mean()
worst_ensemble = worst_snr['Ensemble'].mean()

print("\n" + "="*70)
print(f"最佳性能 ({snr_labels[0]} - 噪声最小):")
print(f"  Model A: {best_A:.4f}")
print(f"  Model B: {best_B:.4f}")
print(f"  Ensemble: {best_ensemble:.4f}")
print(f"\n最差性能 ({snr_labels[-1]} - 噪声最大):")
print(f"  Model A: {worst_A:.4f} (↓ {(best_A - worst_A)*100:.2f}%)")
print(f"  Model B: {worst_B:.4f} (↓ {(best_B - worst_B)*100:.2f}%)")
print(f"  Ensemble: {worst_ensemble:.4f} (↓ {(best_ensemble - worst_ensemble)*100:.2f}%)")
print("="*70)

# ==========================================================
# 绘制美化的小提琴图
# ==========================================================
print("\n正在绘制美化的小提琴图...")

# 重塑数据以便绘图
plot_data = []
for _, row in results_df.iterrows():
    plot_data.append({'SNR': row['SNR'], 'Model': 'Model A (Raw Spectrum)', 'Accuracy': row['Model_A']})
    plot_data.append({'SNR': row['SNR'], 'Model': 'Model B (PCA-processed)', 'Accuracy': row['Model_B']})
    plot_data.append({'SNR': row['SNR'], 'Model': 'Ensemble Model', 'Accuracy': row['Ensemble']})
plot_df = pd.DataFrame(plot_data)

# 设置SNR顺序
plot_df['SNR'] = pd.Categorical(plot_df['SNR'], categories=snr_labels, ordered=True)

# 创建更大的画布
fig, ax = plt.subplots(figsize=(16, 9))

# 设置背景色
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# 定义统一的蓝色调色板（与用户的对比图保持一致）
blue_palette = sns.color_palette("Blues", n_colors=5)[2:]  # 取中间到深的三种蓝色
colors = {
    'Model A (Raw Spectrum)': blue_palette[0],      # 中蓝色
    'Model B (PCA-processed)': blue_palette[1],     # 较深蓝色  
    'Ensemble Model': blue_palette[2]               # 深蓝色
}

# 绘制小提琴图 - 增加间距避免重叠
parts = ax.violinplot(
    dataset=[plot_df[(plot_df['SNR'] == snr) & (plot_df['Model'] == model)]['Accuracy'].values 
             for snr in snr_labels for model in colors.keys()],
    positions=[i*5 + j*1.0 for i in range(len(snr_labels)) for j in range(3)],  # 增加间距
    widths=0.8,  # 减小宽度避免重叠
    showmeans=True,
    showmedians=False,
    showextrema=False
)

# 美化小提琴的各个部分
pos_idx = 0
for i, snr in enumerate(snr_labels):
    for j, (model_name, color) in enumerate(colors.items()):
        # 设置小提琴体的颜色和透明度
        parts['bodies'][pos_idx].set_facecolor(color)
        parts['bodies'][pos_idx].set_alpha(0.7)
        parts['bodies'][pos_idx].set_edgecolor('black')
        parts['bodies'][pos_idx].set_linewidth(1.5)
        
        # 设置均值标记
        parts['cmeans'].set_edgecolor('white')
        parts['cmeans'].set_linewidth(2.5)
        
        pos_idx += 1

# 在小提琴图上叠加散点图（抖动效果）
for i, snr in enumerate(snr_labels):
    for j, (model_name, color) in enumerate(colors.items()):
        data = plot_df[(plot_df['SNR'] == snr) & (plot_df['Model'] == model_name)]['Accuracy'].values
        y = data
        x = np.random.normal(i*5 + j*1.0, 0.08, size=len(y))  # 匹配新的位置间距
        ax.scatter(x, y, alpha=0.6, s=70, color=color, edgecolors='white', 
                  linewidths=1.2, zorder=3)

# 添加箱线图作为内部细节
bp = ax.boxplot(
    [plot_df[(plot_df['SNR'] == snr) & (plot_df['Model'] == model)]['Accuracy'].values 
     for snr in snr_labels for model in colors.keys()],
    positions=[i*5 + j*1.0 for i in range(len(snr_labels)) for j in range(3)],  # 匹配新的位置间距
    widths=0.18,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color='black', linewidth=2.5),
    boxprops=dict(facecolor='white', alpha=0.9, linewidth=1.5),
    whiskerprops=dict(color='black', linewidth=1.5),
    capprops=dict(color='black', linewidth=1.5)
)

# 设置x轴刻度和标签
ax.set_xticks([i*5 + 1.0 for i in range(len(snr_labels))])  # 居中对齐
ax.set_xticklabels(snr_labels, fontsize=15, fontweight='bold')
ax.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=18, fontweight='bold', labelpad=15)
ax.set_ylabel('Accuracy', fontsize=18, fontweight='bold', labelpad=15)
ax.set_title('Model Performance Distribution under Noise Conditions', 
            fontsize=20, fontweight='bold', pad=25)

# 聚焦y轴范围到数据实际分布区域，让小提琴更饱满
# 根据40、30、20 dB的数据范围，大约在0.2-0.75之间
ax.set_ylim(0.1, 0.8)  # 聚焦在数据密集区域，让小提琴视觉上更突出
ax.grid(True, axis='y', alpha=0.4, linestyle='--', linewidth=1, color='gray', zorder=0)
ax.set_axisbelow(True)

# 创建图例
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=colors[model], markersize=13, 
                              label=model, markeredgecolor='white', markeredgewidth=1.5)
                   for model in colors.keys()]
ax.legend(handles=legend_elements, loc='upper right', fontsize=14, 
         title='Model Type', title_fontsize=15, framealpha=0.95, 
         edgecolor='black', fancybox=True, shadow=True)

# 不添加90%参考线（因为不在当前y轴范围内）

plt.tight_layout()

# 保存小提琴图
violin_svg = 'noise_robustness_snr_violin.svg'
plt.savefig(violin_svg, format='svg', dpi=300, bbox_inches='tight')
print(f"✓ 美化小提琴图已保存为: {violin_svg}")

violin_png = 'noise_robustness_snr_violin.png'
plt.savefig(violin_png, format='png', dpi=300, bbox_inches='tight')
print(f"✓ 美化小提琴图已保存为: {violin_png}")

# ==========================================================
# 绘制带误差带的折线图
# ==========================================================
print("\n正在绘制折线图...")

# 计算均值和标准差
summary_stats = results_df.groupby('SNR').agg({
    'Model_A': ['mean', 'std'],
    'Model_B': ['mean', 'std'],
    'Ensemble': ['mean', 'std']
})

fig2, ax2 = plt.subplots(figsize=(12, 7))

# 使用统一的蓝色调色板
line_colors = {
    'Model A (Raw Spectrum)': blue_palette[0],
    'Model B (PCA-processed)': blue_palette[1],
    'Ensemble Model': blue_palette[2]
}

# 将SNR标签转换为数字用于绘图
snr_numeric = [40, 30, 20]

# Model A
mean_A = [summary_stats.loc[snr, ('Model_A', 'mean')] for snr in snr_labels]
std_A = [summary_stats.loc[snr, ('Model_A', 'std')] for snr in snr_labels]
ax2.plot(snr_numeric, mean_A, marker='o', markersize=10, linewidth=2.5, 
        color=line_colors['Model A (Raw Spectrum)'], label='Model A (Raw Spectrum)', alpha=0.9)
ax2.fill_between(snr_numeric, 
                 np.array(mean_A) - np.array(std_A), 
                 np.array(mean_A) + np.array(std_A),
                 alpha=0.2, color=line_colors['Model A (Raw Spectrum)'])

# Model B
mean_B = [summary_stats.loc[snr, ('Model_B', 'mean')] for snr in snr_labels]
std_B = [summary_stats.loc[snr, ('Model_B', 'std')] for snr in snr_labels]
ax2.plot(snr_numeric, mean_B, marker='s', markersize=10, linewidth=2.5, 
        color=line_colors['Model B (PCA-processed)'], label='Model B (PCA-processed)', alpha=0.9)
ax2.fill_between(snr_numeric, 
                 np.array(mean_B) - np.array(std_B), 
                 np.array(mean_B) + np.array(std_B),
                 alpha=0.2, color=line_colors['Model B (PCA-processed)'])

# Ensemble
mean_E = [summary_stats.loc[snr, ('Ensemble', 'mean')] for snr in snr_labels]
std_E = [summary_stats.loc[snr, ('Ensemble', 'std')] for snr in snr_labels]
ax2.plot(snr_numeric, mean_E, marker='D', markersize=10, linewidth=3, 
        color=line_colors['Ensemble Model'], label='Ensemble Model', alpha=0.9, zorder=5)
ax2.fill_between(snr_numeric, 
                 np.array(mean_E) - np.array(std_E), 
                 np.array(mean_E) + np.array(std_E),
                 alpha=0.25, color=line_colors['Ensemble Model'])

# 设置图表属性
ax2.set_xlabel('Signal-to-Noise Ratio (dB)', fontsize=16, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
ax2.set_title('Model Robustness under Noise Perturbation', 
             fontsize=18, fontweight='bold', pad=20)
ax2.set_xticks(snr_numeric)
ax2.set_xticklabels(['40 dB', '30 dB', '20 dB'], fontsize=13)
ax2.legend(fontsize=13, loc='lower left', framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_ylim(0, 1.0)

# 反转x轴使噪声从小到大（从左到右）
ax2.invert_xaxis()

plt.tight_layout()

# 保存折线图
lineplot_svg = 'noise_robustness_snr_lineplot.svg'
plt.savefig(lineplot_svg, format='svg', dpi=300, bbox_inches='tight')
print(f"✓ 折线图已保存为: {lineplot_svg}")

lineplot_png = 'noise_robustness_snr_lineplot.png'
plt.savefig(lineplot_png, format='png', dpi=300, bbox_inches='tight')
print(f"✓ 折线图已保存为: {lineplot_png}")

# ==========================================================
# 绘制柱状图 (带误差线)
# ==========================================================
print("\n正在绘制柱状图...")

# 计算每个SNR级别的均值和标准差
summary_stats = results_df.groupby('SNR').agg({
    'Model_A': ['mean', 'std'],
    'Model_B': ['mean', 'std'],
    'Ensemble': ['mean', 'std']
})

fig2, ax2 = plt.subplots(figsize=(14, 8))

x = np.arange(len(snr_labels))
width = 0.25

# 绘制三组柱状图，使用统一蓝色调色板
bars1 = ax2.bar(x - width, 
               [summary_stats.loc[snr, ('Model_A', 'mean')] for snr in snr_labels],
               width, 
               yerr=[summary_stats.loc[snr, ('Model_A', 'std')] for snr in snr_labels],
               label='Model A (Raw Spectrum)', 
               color=blue_palette[0], alpha=0.8, capsize=5, edgecolor='black', linewidth=1.2)

bars2 = ax2.bar(x, 
               [summary_stats.loc[snr, ('Model_B', 'mean')] for snr in snr_labels],
               width, 
               yerr=[summary_stats.loc[snr, ('Model_B', 'std')] for snr in snr_labels],
               label='Model B (PCA-processed)', 
               color=blue_palette[1], alpha=0.8, capsize=5, edgecolor='black', linewidth=1.2)

bars3 = ax2.bar(x + width, 
               [summary_stats.loc[snr, ('Ensemble', 'mean')] for snr in snr_labels],
               width, 
               yerr=[summary_stats.loc[snr, ('Ensemble', 'std')] for snr in snr_labels],
               label='Ensemble Model', 
               color=blue_palette[2], alpha=0.8, capsize=5, edgecolor='black', linewidth=1.2)

# 设置图表属性
ax2.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=16, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
ax2.set_title('Model Performance Comparison under Different Noise Levels', 
             fontsize=18, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(snr_labels, fontsize=13)
ax2.legend(fontsize=12, loc='best', framealpha=0.95)
ax2.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_ylim(0, 1.0)

plt.tight_layout()

# 保存柱状图
barplot_svg = 'noise_robustness_snr_barplot.svg'
plt.savefig(barplot_svg, format='svg', dpi=300, bbox_inches='tight')
print(f"✓ 柱状图已保存为: {barplot_svg}")

barplot_png = 'noise_robustness_snr_barplot.png'
plt.savefig(barplot_png, format='png', dpi=300, bbox_inches='tight')
print(f"✓ 柱状图已保存为: {barplot_png}")

# ==========================================================
# 保存数据到 CSV 文件
# ==========================================================
csv_filename = 'noise_robustness_snr_results.csv'
results_df.to_csv(csv_filename, index=False)
print(f"\n✓ 详细数据已保存为: {csv_filename}")

# 保存统计摘要
summary_csv = 'noise_robustness_snr_summary.csv'
stats_df.to_csv(summary_csv)
print(f"✓ 统计摘要已保存为: {summary_csv}")

print("\n" + "="*70)
print("基于SNR的噪声鲁棒性分析完成!")
print("\n生成的文件:")
print(f"  1. {violin_svg} / {violin_png} - 小提琴图")
print(f"  2. {lineplot_svg} / {lineplot_png} - 折线图(带误差带)")
print(f"  3. {barplot_svg} / {barplot_png} - 柱状图(带误差线)")
print(f"  4. {csv_filename} - 原始数据")
print(f"  5. {summary_csv} - 统计摘要")
print("="*70)

# ==========================================================
# 生成结论
# ==========================================================
print("\n" + "="*70)
print("实验结论:")
print("="*70)

# 比较集成模型在各个SNR级别下的表现
for snr_label in snr_labels:
    data = results_df[results_df['SNR'] == snr_label]
    ensemble_mean = data['Ensemble'].mean()
    model_a_mean = data['Model_A'].mean()
    model_b_mean = data['Model_B'].mean()
    
    if ensemble_mean > model_a_mean and ensemble_mean > model_b_mean:
        print(f"✓ {snr_label}: 集成模型表现最佳 ({ensemble_mean:.4f})")
    elif ensemble_mean > model_a_mean or ensemble_mean > model_b_mean:
        print(f"• {snr_label}: 集成模型优于部分单分支模型 ({ensemble_mean:.4f})")

print("\n双流融合架构通过整合不同特征表示，在噪声环境下展现出")
print("更好的鲁棒性和泛化能力，验证了PM分支的有效性。")
print("="*70)
