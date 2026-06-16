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
_, X_eval_orig, _, y_eval = train_test_split(
    X_processed, y_orig, test_size=0.3, random_state=42, stratify=y_orig
)

# 分割评估集为调参集和测试集
print(f"\n将评估集分割为调参集和测试集...")
X_tune_orig, X_test_orig, y_tune, y_test = train_test_split(
    X_eval_orig, y_eval, test_size=0.5, random_state=42, stratify=y_eval
)
print(f"调参集 (Tuning Set) 大小: {len(y_tune)}")
print(f"最终测试集 (Test Set) 大小: {len(y_test)}")

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
except Exception as e:
    print(f"!!! 错误: 加载模型失败 !!!")
    print(f"原始错误: {e}")
    exit()

# --- 准备数据 (Test Set) ---
print("\n准备测试集数据...")
X_test_raw = scaler_A.transform(X_test_orig)
X_test_raw_cnn = X_test_raw.reshape(X_test_raw.shape[0], N_BANDS_RAW, 1)
X_test_scaled_B = scaler_B.transform(X_test_orig)
X_test_pca = pca_B.transform(X_test_scaled_B)
X_test_pca_cnn = X_test_pca.reshape(X_test_pca.shape[0], N_COMPONENTS_PCA, 1)

# --- 获取测试集上的预测概率 ---
print("\n获取两个模型在测试集上的预测概率...")
probs_A_test = model_A.predict(X_test_raw_cnn)
probs_B_test = model_B.predict(X_test_pca_cnn)

# ==========================================================
# 核心部分：测试权重从 0.0 到 1.0
# ==========================================================
print("\n" + "="*60)
print("开始测试集成权重从 0.0 到 1.0 的准确率变化...")
print("="*60 + "\n")

# 定义权重范围（从0到1，步长0.01）
weight_range = np.arange(0.0, 1.01, 0.01)
accuracies = []

for w_A in tqdm(weight_range, desc="Testing weights"):
    w_B = 1.0 - w_A
    
    # 融合概率
    final_probs = (w_A * probs_A_test) + (w_B * probs_B_test)
    y_pred = np.argmax(final_probs, axis=1)
    
    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
# 转换为 numpy 数组以便分析
accuracies = np.array(accuracies)

# --- 统计分析 ---
print("\n" + "="*60)
print("融合策略稳定性分析结果")
print("="*60)
print(f"最高准确率: {accuracies.max():.4f} (权重 Model A = {weight_range[accuracies.argmax()]:.2f})")
print(f"最低准确率: {accuracies.min():.4f} (权重 Model A = {weight_range[accuracies.argmin()]:.2f})")
print(f"平均准确率: {accuracies.mean():.4f}")
print(f"准确率标准差: {accuracies.std():.4f}")
print(f"准确率范围: {accuracies.max() - accuracies.min():.4f}")
print("="*60 + "\n")

# 分析准确率 >= 90% 的权重范围
high_acc_indices = np.where(accuracies >= 0.90)[0]
if len(high_acc_indices) > 0:
    print(f"准确率 >= 90% 的权重范围:")
    print(f"  Model A 权重: [{weight_range[high_acc_indices[0]]:.2f}, {weight_range[high_acc_indices[-1]]:.2f}]")
    print(f"  覆盖范围: {len(high_acc_indices)} 个权重值 ({len(high_acc_indices)/len(weight_range)*100:.1f}%)")
    print()

# ==========================================================
# 绘制折线图
# ==========================================================
print("正在绘制权重-准确率折线图...")

fig, ax = plt.subplots(figsize=(12, 7))

# 主折线图
ax.plot(weight_range, accuracies, linewidth=2.5, color='#2E86AB', label='Accuracy')

# 标记最大值和最小值
max_idx = accuracies.argmax()
min_idx = accuracies.argmin()
ax.scatter(weight_range[max_idx], accuracies[max_idx], 
          color='#06A77D', s=150, zorder=5, marker='o', 
          edgecolors='black', linewidths=1.5,
          label=f'Max: {accuracies[max_idx]:.4f} (w={weight_range[max_idx]:.2f})')
ax.scatter(weight_range[min_idx], accuracies[min_idx], 
          color='#D81E5B', s=150, zorder=5, marker='s', 
          edgecolors='black', linewidths=1.5,
          label=f'Min: {accuracies[min_idx]:.4f} (w={weight_range[min_idx]:.2f})')

# 添加水平参考线（90%准确率）
ax.axhline(y=0.90, color='#F77F00', linestyle='--', linewidth=1.5, 
          alpha=0.7, label='90% Threshold')

# 填充高准确率区域
if len(high_acc_indices) > 0:
    ax.fill_between(weight_range, 0.85, accuracies, 
                    where=(accuracies >= 0.90), 
                    alpha=0.2, color='#06A77D', 
                    label='Accuracy ≥ 90%')

# 设置图表属性
ax.set_xlabel('Weight of Model A (Model B = 1 - Weight)', fontsize=16, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
ax.set_title('Ensemble Model Stability Analysis: Accuracy vs. Weight', 
            fontsize=18, fontweight='bold', pad=20)
ax.set_xlim(0, 1)
ax.set_ylim(0.85, max(accuracies.max() + 0.02, 0.95))
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.legend(loc='best', fontsize=12, framealpha=0.95)

# 添加次坐标轴显示 Model B 权重
ax2 = ax.twiny()
ax2.set_xlabel('Weight of Model B (Model A = 1 - Weight)', fontsize=16, fontweight='bold')
ax2.set_xlim(1, 0)

plt.tight_layout()

# 保存图表
output_filename = 'ensemble_weight_stability_analysis.svg'
plt.savefig(output_filename, format='svg', dpi=300, bbox_inches='tight')
print(f"\n折线图已保存为: {output_filename}")

output_filename_png = 'ensemble_weight_stability_analysis.png'
plt.savefig(output_filename_png, format='png', dpi=300, bbox_inches='tight')
print(f"折线图已保存为: {output_filename_png}")

# ==========================================================
# 保存数据到 CSV 文件
# ==========================================================
results_df = pd.DataFrame({
    'Weight_Model_A': weight_range,
    'Weight_Model_B': 1.0 - weight_range,
    'Accuracy': accuracies
})

csv_filename = 'ensemble_weight_analysis_results.csv'
results_df.to_csv(csv_filename, index=False)
print(f"详细数据已保存为: {csv_filename}")

print("\n" + "="*60)
print("分析完成!")
print("="*60)
