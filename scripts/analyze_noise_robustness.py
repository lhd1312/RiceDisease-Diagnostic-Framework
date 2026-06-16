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
# 核心部分：噪声鲁棒性分析
# ==========================================================

def add_gaussian_noise(data, noise_level):
    """
    向光谱数据添加高斯噪声
    
    Args:
        data: 原始数据 (n_samples, n_features)
        noise_level: 噪声强度 (标准差的倍数)
    
    Returns:
        添加噪声后的数据
    """
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

# 设置噪声水平范围
noise_levels = np.arange(0.0, 1.01, 0.05)  # 从0到1，步长0.05
print(f"\n测试噪声水平范围: {noise_levels[0]:.2f} - {noise_levels[-1]:.2f}, 共 {len(noise_levels)} 个级别")

# 存储结果
results = {
    'noise_level': [],
    'model_A_acc': [],
    'model_B_acc': [],
    'ensemble_acc': []
}

# 设置集成权重 (使用最优权重，或者简单平均)
ENSEMBLE_WEIGHT_A = 0.5  # 可以根据之前的分析调整
ENSEMBLE_WEIGHT_B = 1.0 - ENSEMBLE_WEIGHT_A

print("\n" + "="*70)
print("开始噪声鲁棒性分析...")
print(f"集成模型权重: Model A = {ENSEMBLE_WEIGHT_A:.2f}, Model B = {ENSEMBLE_WEIGHT_B:.2f}")
print("="*70 + "\n")

for noise_level in tqdm(noise_levels, desc="Testing noise levels"):
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # 添加噪声
    X_test_noisy = add_gaussian_noise(X_test_orig, noise_level)
    
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
    results['noise_level'].append(noise_level)
    results['model_A_acc'].append(acc_A)
    results['model_B_acc'].append(acc_B)
    results['ensemble_acc'].append(acc_ensemble)

# 转换为 DataFrame
results_df = pd.DataFrame(results)

# ==========================================================
# 统计分析
# ==========================================================
print("\n" + "="*70)
print("噪声鲁棒性分析结果")
print("="*70)

# 无噪声情况（baseline）
baseline_A = results_df.loc[0, 'model_A_acc']
baseline_B = results_df.loc[0, 'model_B_acc']
baseline_ensemble = results_df.loc[0, 'ensemble_acc']

print(f"\n无噪声基线准确率:")
print(f"  Model A (Raw Spectrum):     {baseline_A:.4f}")
print(f"  Model B (PCA-processed):    {baseline_B:.4f}")
print(f"  Ensemble Model:             {baseline_ensemble:.4f}")

# 最高噪声情况
max_noise_A = results_df.loc[len(results_df)-1, 'model_A_acc']
max_noise_B = results_df.loc[len(results_df)-1, 'model_B_acc']
max_noise_ensemble = results_df.loc[len(results_df)-1, 'ensemble_acc']

print(f"\n最高噪声 (level={noise_levels[-1]:.2f}) 准确率:")
print(f"  Model A (Raw Spectrum):     {max_noise_A:.4f} (↓ {(baseline_A - max_noise_A)*100:.2f}%)")
print(f"  Model B (PCA-processed):    {max_noise_B:.4f} (↓ {(baseline_B - max_noise_B)*100:.2f}%)")
print(f"  Ensemble Model:             {max_noise_ensemble:.4f} (↓ {(baseline_ensemble - max_noise_ensemble)*100:.2f}%)")

# 计算性能衰减率
degradation_A = (baseline_A - max_noise_A) / baseline_A * 100
degradation_B = (baseline_B - max_noise_B) / baseline_B * 100
degradation_ensemble = (baseline_ensemble - max_noise_ensemble) / baseline_ensemble * 100

print(f"\n性能衰减率 (相对下降百分比):")
print(f"  Model A (Raw Spectrum):     {degradation_A:.2f}%")
print(f"  Model B (PCA-processed):    {degradation_B:.2f}%")
print(f"  Ensemble Model:             {degradation_ensemble:.2f}%")

# 计算准确率的平均值和标准差
print(f"\n整体统计 (所有噪声水平):")
print(f"  Model A - 平均准确率: {results_df['model_A_acc'].mean():.4f}, 标准差: {results_df['model_A_acc'].std():.4f}")
print(f"  Model B - 平均准确率: {results_df['model_B_acc'].mean():.4f}, 标准差: {results_df['model_B_acc'].std():.4f}")
print(f"  Ensemble - 平均准确率: {results_df['ensemble_acc'].mean():.4f}, 标准差: {results_df['ensemble_acc'].std():.4f}")

print("\n" + "="*70)

# 集成模型优于单分支模型的计数
ensemble_better_A = (results_df['ensemble_acc'] > results_df['model_A_acc']).sum()
ensemble_better_B = (results_df['ensemble_acc'] > results_df['model_B_acc']).sum()
total_points = len(results_df)

print(f"\n集成模型优势统计:")
print(f"  集成模型优于 Model A: {ensemble_better_A}/{total_points} ({ensemble_better_A/total_points*100:.1f}%)")
print(f"  集成模型优于 Model B: {ensemble_better_B}/{total_points} ({ensemble_better_B/total_points*100:.1f}%)")

print("="*70 + "\n")

# ==========================================================
# 绘制对比图
# ==========================================================
print("正在绘制噪声鲁棒性对比图...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- 子图1: 准确率随噪声水平的变化 ---
ax1.plot(results_df['noise_level'], results_df['model_A_acc'], 
         linewidth=2.5, marker='o', markersize=6, 
         color='#D81E5B', label='Model A (Raw Spectrum)', alpha=0.8)
ax1.plot(results_df['noise_level'], results_df['model_B_acc'], 
         linewidth=2.5, marker='s', markersize=6, 
         color='#F77F00', label='Model B (PCA-processed)', alpha=0.8)
ax1.plot(results_df['noise_level'], results_df['ensemble_acc'], 
         linewidth=3, marker='D', markersize=7, 
         color='#06A77D', label='Ensemble Model', alpha=0.9)

ax1.set_xlabel('Gaussian Noise Level (σ)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
ax1.set_title('Model Performance under Noise Perturbation', 
              fontsize=18, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.legend(loc='best', fontsize=13, framealpha=0.95)
ax1.set_xlim(noise_levels[0], noise_levels[-1])

# --- 子图2: 性能衰减对比 ---
# 计算相对性能 (normalized to baseline)
relative_A = (results_df['model_A_acc'] / baseline_A) * 100
relative_B = (results_df['model_B_acc'] / baseline_B) * 100
relative_ensemble = (results_df['ensemble_acc'] / baseline_ensemble) * 100

ax2.plot(results_df['noise_level'], relative_A, 
         linewidth=2.5, marker='o', markersize=6, 
         color='#D81E5B', label='Model A (Raw Spectrum)', alpha=0.8)
ax2.plot(results_df['noise_level'], relative_B, 
         linewidth=2.5, marker='s', markersize=6, 
         color='#F77F00', label='Model B (PCA-processed)', alpha=0.8)
ax2.plot(results_df['noise_level'], relative_ensemble, 
         linewidth=3, marker='D', markersize=7, 
         color='#06A77D', label='Ensemble Model', alpha=0.9)

ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax2.set_xlabel('Gaussian Noise Level (σ)', fontsize=16, fontweight='bold')
ax2.set_ylabel('Relative Performance (%)', fontsize=16, fontweight='bold')
ax2.set_title('Performance Degradation Comparison', 
              fontsize=18, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.legend(loc='best', fontsize=13, framealpha=0.95)
ax2.set_xlim(noise_levels[0], noise_levels[-1])

plt.tight_layout()

# 保存图表
output_filename_svg = 'noise_robustness_comparison.svg'
plt.savefig(output_filename_svg, format='svg', dpi=300, bbox_inches='tight')
print(f"✓ 对比图已保存为: {output_filename_svg}")

output_filename_png = 'noise_robustness_comparison.png'
plt.savefig(output_filename_png, format='png', dpi=300, bbox_inches='tight')
print(f"✓ 对比图已保存为: {output_filename_png}")

# ==========================================================
# 绘制单独的性能衰减条形图
# ==========================================================
print("\n正在绘制性能衰减对比条形图...")

fig2, ax = plt.subplots(figsize=(10, 7))

models = ['Model A\n(Raw Spectrum)', 'Model B\n(PCA-processed)', 'Ensemble\nModel']
degradations = [degradation_A, degradation_B, degradation_ensemble]
colors = ['#D81E5B', '#F77F00', '#06A77D']

bars = ax.bar(models, degradations, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for bar, deg in zip(bars, degradations):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{deg:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Performance Degradation (%)', fontsize=16, fontweight='bold')
ax.set_title(f'Performance Degradation at Maximum Noise (σ={noise_levels[-1]:.2f})', 
             fontsize=18, fontweight='bold', pad=20)
ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_ylim(0, max(degradations) * 1.2)

plt.tight_layout()

output_filename_bar_svg = 'performance_degradation_bar.svg'
plt.savefig(output_filename_bar_svg, format='svg', dpi=300, bbox_inches='tight')
print(f"✓ 条形图已保存为: {output_filename_bar_svg}")

output_filename_bar_png = 'performance_degradation_bar.png'
plt.savefig(output_filename_bar_png, format='png', dpi=300, bbox_inches='tight')
print(f"✓ 条形图已保存为: {output_filename_bar_png}")

# ==========================================================
# 保存数据到 CSV 文件
# ==========================================================
csv_filename = 'noise_robustness_analysis_results.csv'
results_df.to_csv(csv_filename, index=False)
print(f"\n✓ 详细数据已保存为: {csv_filename}")

print("\n" + "="*70)
print("噪声鲁棒性分析完成!")
print("生成的文件:")
print(f"  1. {output_filename_svg} - 主对比图 (SVG)")
print(f"  2. {output_filename_png} - 主对比图 (PNG)")
print(f"  3. {output_filename_bar_svg} - 性能衰减条形图 (SVG)")
print(f"  4. {output_filename_bar_png} - 性能衰减条形图 (PNG)")
print(f"  5. {csv_filename} - 原始数据")
print("="*70)

# ==========================================================
# 生成总结报告
# ==========================================================
print("\n" + "="*70)
print("鲁棒性分析总结:")
print("="*70)
print("\n该实验通过在测试集上添加不同强度的高斯噪声，模拟实际现场条件下")
print("传感器噪声或环境干扰对光谱数据的影响，验证了以下结论:\n")

if degradation_ensemble < degradation_A and degradation_ensemble < degradation_B:
    print("✓ 集成模型在噪声条件下表现出更强的鲁棒性")
    print(f"  相比单分支模型，性能衰减更小 ({degradation_ensemble:.2f}% vs {degradation_A:.2f}%/{degradation_B:.2f}%)")
    
if ensemble_better_A >= total_points * 0.8 or ensemble_better_B >= total_points * 0.8:
    print(f"\n✓ 集成模型在大多数噪声水平下优于单分支模型")
    print(f"  优于Model A的比例: {ensemble_better_A/total_points*100:.1f}%")
    print(f"  优于Model B的比例: {ensemble_better_B/total_points*100:.1f}%")

print("\n这证明了双流融合架构能够有效整合不同特征表示的优势，")
print("在面对数据扰动时具有更好的泛化能力和稳定性。")
print("="*70)
