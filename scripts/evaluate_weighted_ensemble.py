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
# VVVV --- 1. 导入自定义模块所需的层 --- VVVV
from tensorflow.keras.layers import (
    Dense, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, 
    Reshape, Multiply, Concatenate, Add, Activation, 
    SeparableConv1D, BatchNormalization
)
# ^^^^ ---------------------------------- ^^^^
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- Matplotlib 字体设置 (保持不变) ---
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 18,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'xtick.minor.width': 1.5, 'ytick.minor.width': 1.5, 'xtick.direction': 'in',
    'ytick.direction': 'in'
})

# ==========================================================
# VVVV --- 2. 添加自定义模块定义 (加载模型所必需) --- VVVV
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

# --- 数据路径和常量 (保持不变) ---
DATA_DIR = Path('DATA'); VNIR_FILE = DATA_DIR / 'VNIR.xlsx'; SWIR_FILE = DATA_DIR / 'SWIR.xlsx'
SHEET_NAMES = [
    'CK', 'DWB_01_Resistant', 'DWB_02_Resistant', 'DWB_03_Resistant', 'DWB_04_Resistant',
    'DWB_05_Susceptible', 'BYK_01_Resistant', 'BYK_02_Resistant', 'BYK_03_Resistant', 'BYK_04_Susceptible'
]
label_map = {name: i for i, name in enumerate(SHEET_NAMES)}
N_CLASSES = 10; N_COMPONENTS_PCA = 100

# --- 数据加载 (保持不变) ---
features = []; labels = []; all_bands = np.array([])
print("开始加载和合并数据 (Ensemble)...")
with pd.ExcelFile(VNIR_FILE) as vnir_xls, pd.ExcelFile(SWIR_FILE) as swir_xls:
    for sheet in tqdm(SHEET_NAMES, desc="Processing sheets"):
        vnir_data = pd.read_excel(vnir_xls, sheet_name=sheet, header=0)
        swir_data = pd.read_excel(swir_xls, sheet_name=sheet, header=0)
        if not all_bands.any():
            vnir_bands = vnir_data.columns.astype(float); swir_bands = swir_data.columns.astype(float)
            all_bands = np.concatenate([vnir_bands, swir_bands])
        combined_data = pd.concat([vnir_data, swir_data], axis=1); features.append(combined_data)
        current_labels = np.full(combined_data.shape[0], label_map[sheet]); labels.append(current_labels)
X_orig = pd.concat(features, ignore_index=True).values; y_orig = np.concatenate(labels)

# --- 裁剪 (保持不变) ---
WAVELENGTH_RANGES_TO_KEEP = [(420, 900), (1000, 1700)]
idx_keep = np.zeros(all_bands.shape, dtype=bool)
for (start, end) in WAVELENGTH_RANGES_TO_KEEP:
    idx_keep = idx_keep | ((all_bands >= start) & (all_bands <= end))
X_orig = X_orig[:, idx_keep]
N_BANDS_RAW = X_orig.shape[1]

# --- 预处理 (保持不变) ---
print("\n开始光谱预处理 (SG+SNV)...")
X_processed = savgol_filter(X_orig, window_length=11, polyorder=3, axis=1)
mean_row = np.mean(X_processed, axis=1, keepdims=True)
std_row = np.std(X_processed, axis=1, keepdims=True)
X_processed = (X_processed - mean_row) / (std_row + 1e-8)

# --- 划分 "评估集" (30%) ---
# 我们假设模型是在另外 70% 的数据上训练的
_, X_eval_orig, _, y_eval = train_test_split(
    X_processed, y_orig, test_size=0.3, random_state=42, stratify=y_orig
)

# VVVV --- 3. 再次分割评估集：Tuning (调参) vs Test (评估) --- VVVV
# 我们必须分割 "评估集" (X_eval_orig)，
# 否则在寻找最佳权重时会发生 "数据泄露"
print(f"\n将 30% 评估集 (n={len(y_eval)}) 再次分割...")
X_tune_orig, X_test_orig, y_tune, y_test = train_test_split(
    X_eval_orig, y_eval, test_size=0.5, random_state=42, stratify=y_eval
)
print(f"调参集 (Tuning Set) 大小: {len(y_tune)}")
print(f"最终测试集 (Test Set) 大小: {len(y_test)}")
# ^^^^ ---------------------------------------------------- ^^^^

# --- 加载模型和预处理器 ---
print("\n加载所有模型和预处理器...")
try:
    # VVVV --- 4. 使用 custom_objects 加载模型 (!!! 关键修复 !!!) --- VVVV
    custom_objects = {
        "cbam_block": cbam_block,
        "multiscale_block": multiscale_block
    }
    # (注意：我假设 Model A 是 msc_cbam, Model B 是 msc)
    model_A = load_model("Ablation_msc_cbam.h5", custom_objects=custom_objects)
    model_B = load_model("Model_B_msc.h5", custom_objects=custom_objects)
    
    scaler_A = joblib.load("scaler_A_trimmed.pkl")
    scaler_B = joblib.load("scaler_B_trimmed.pkl")
    pca_B = joblib.load("pca_B_trimmed.pkl")
    # ^^^^ ----------------------------------------------------- ^^^^
except Exception as e:
    print(f"!!! 错误: 加载模型失败 !!!"); print(f"原始错误: {e}"); exit()

# --- 准备数据 (Tuning Set) ---
X_tune_raw = scaler_A.transform(X_tune_orig)
X_tune_raw_cnn = X_tune_raw.reshape(X_tune_raw.shape[0], N_BANDS_RAW, 1)
X_tune_scaled_B = scaler_B.transform(X_tune_orig)
X_tune_pca = pca_B.transform(X_tune_scaled_B)
X_tune_pca_cnn = X_tune_pca.reshape(X_tune_pca.shape[0], N_COMPONENTS_PCA, 1)
print("Tuning (调参) 数据准备完毕。")

# --- 准备数据 (Test Set) ---
X_test_raw = scaler_A.transform(X_test_orig)
X_test_raw_cnn = X_test_raw.reshape(X_test_raw.shape[0], N_BANDS_RAW, 1)
X_test_scaled_B = scaler_B.transform(X_test_orig)
X_test_pca = pca_B.transform(X_test_scaled_B)
X_test_pca_cnn = X_test_pca.reshape(X_test_pca.shape[0], N_COMPONENTS_PCA, 1)
print("Test (最终评估) 数据准备完毕。")

# --- 获取 Tuning Set 上的概率 ---
print("\n在 Tuning Set 上获取两个模型的预测概率...")
probs_A_tune = model_A.predict(X_tune_raw_cnn)
probs_B_tune = model_B.predict(X_tune_pca_cnn)

# VVVV --- 5. 寻找最佳权重 (Weighted Average Search) --- VVVV
print("正在 Tuning Set 上搜索最佳集成权重...")
best_acc = 0
best_w_A = 0.5
search_steps = np.arange(0.1, 1.0, 0.05) # 搜索步长 0.05

for w_A in search_steps:
    w_B = 1.0 - w_A
    
    # 融合概率
    final_probs = (w_A * probs_A_tune) + (w_B * probs_B_tune)
    y_pred_tune = np.argmax(final_probs, axis=1)
    
    acc = accuracy_score(y_tune, y_pred_tune)
    
    if acc > best_acc:
        best_acc = acc
        best_w_A = w_A

print(f"\n--- 最佳权重搜索完毕 ---")
print(f"在 Tuning Set 上的最佳准确率: {best_acc:.4f}")
print(f"最佳权重 (Model A): {best_w_A:.2f}")
print(f"最佳权重 (Model B): {(1.0 - best_w_A):.2f}")
# ^^^^ -------------------------------------------------- ^^^^

# --- 6. 在最终 Test Set 上应用最佳权重 ---
print("\n在最终 Test Set 上应用最佳权重进行评估...")
probs_A_test = model_A.predict(X_test_raw_cnn)
probs_B_test = model_B.predict(X_test_pca_cnn)

final_probs_test = (best_w_A * probs_A_test) + ((1.0 - best_w_A) * probs_B_test)
y_pred_test = np.argmax(final_probs_test, axis=1)

# --- 7. 评估最终结果 ---
accuracy = accuracy_score(y_test, y_pred_test)
print(f"\n最终 [加权集成] 模型在「最终测试集」上的表现:")
print(f"测试集 准确率 (Accuracy): {accuracy:.4f}")

if accuracy >= 0.90:
    print("\n**************************************")
    print(f"   恭喜！模型准确率 {accuracy:.4f} 达到了90%以上的目标！")
    print("**************************************")
else:
    print(f"\n模型准确率 {accuracy:.4f}。")

# 分类报告
print("\n分类报告 (Classification Report):")
print(classification_report(y_test, y_pred_test, target_names=SHEET_NAMES, digits=4))

# 绘制混淆矩阵
print("正在绘制混淆矩阵并保存为 confusion_matrix_ensemble_WEIGHTED.svg ...")
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=SHEET_NAMES, yticklabels=SHEET_NAMES)
plt.title(f"Weighted Ensemble (A={best_w_A:.2f}, B={(1.0-best_w_A):.2f}) (Acc: {accuracy:.4f})")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=90); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_ensemble_WEIGHTED.svg", format='svg')
print("\n--- 任务完成 ---")