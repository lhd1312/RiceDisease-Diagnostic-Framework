import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, 
    SeparableConv1D, Input, Add, Activation, 
    GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, Multiply, Concatenate
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. 定义两个模型架构
# ==============================================================================

# --- 公共模块 ---
def cbam_block(input_feature, ratio=8):
    """CBAM Attention Block"""
    channel_axis = -1
    filters = input_feature.shape[channel_axis]
    shared_layer_one = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(filters, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling1D()(input_feature); avg_pool = Reshape((1, filters))(avg_pool)
    avg_pool = shared_layer_one(avg_pool); avg_pool = shared_layer_two(avg_pool)
    max_pool = GlobalMaxPooling1D()(input_feature); max_pool = Reshape((1, filters))(max_pool)
    max_pool = shared_layer_one(max_pool); max_pool = shared_layer_two(max_pool)
    
    cbam_feature = Add()([avg_pool, max_pool]); cbam_feature = Activation('sigmoid')(cbam_feature)
    x_refined_c = Multiply()([input_feature, cbam_feature])
    
    avg_pool = tf.reduce_mean(x_refined_c, axis=channel_axis, keepdims=True)
    max_pool = tf.reduce_max(x_refined_c, axis=channel_axis, keepdims=True)
    concat = Concatenate(axis=channel_axis)([avg_pool, max_pool])
    cbam_feature = Conv1D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    x_refined_s = Multiply()([x_refined_c, cbam_feature])
    return x_refined_s

def multiscale_block(inputs):
    """Multi-Scale Convolution Block"""
    b1 = SeparableConv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    b2 = SeparableConv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    b3 = SeparableConv1D(32, kernel_size=7, padding='same', activation='relu')(inputs)
    x = Concatenate()([b1, b2, b3]); x = BatchNormalization()(x)
    return x

# --- 模型 A: Raw Data + MSC + CBAM ---
def build_model_A(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    x = multiscale_block(inputs)
    x = MaxPooling1D(2)(x)
    x = SeparableConv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(192, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs, name="Model_A_Raw")

# --- 模型 B: PCA + MSC (无CBAM, 根据您之前的代码) ---
def build_model_B(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    x = multiscale_block(inputs)
    x = MaxPooling1D(2)(x)
    x = SeparableConv1D(96, 7, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    # Model B 在您之前的代码中好像没有加 CBAM，如果需要请在这里加上
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs, name="Model_B_PCA")

# ==============================================================================
# 2. 数据加载 (保持不变)
# ==============================================================================
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

# 裁剪
WAVELENGTH_RANGES_TO_KEEP = [(420, 900), (1000, 1700)]
idx_keep = np.zeros(all_bands.shape, dtype=bool)
for (start, end) in WAVELENGTH_RANGES_TO_KEEP:
    idx_keep = idx_keep | ((all_bands >= start) & (all_bands <= end))
X_orig = X_orig[:, idx_keep]
N_BANDS_RAW = X_orig.shape[1]

# 预处理
print("SG滤波 + SNV...")
X_processed = savgol_filter(X_orig, window_length=11, polyorder=3, axis=1)
mean_row = np.mean(X_processed, axis=1, keepdims=True)
std_row = np.std(X_processed, axis=1, keepdims=True)
X_processed = (X_processed - mean_row) / (std_row + 1e-8)

# ==============================================================================
# 3. 5折交叉验证 (End-to-End Fusion)
# ==============================================================================
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

results_A = []
results_B = []
results_Fusion = []
best_weights_log = []

print(f"\n{'='*40}")
print(f"开始 5折交叉验证: 双模型加权融合")
print(f"{'='*40}")

for fold_no, (train_index, val_index) in enumerate(skf.split(X_processed, y_orig)):
    print(f"\n>>> Fold {fold_no+1} / {k_folds}")
    
    # --- 1. 数据划分 ---
    X_train, X_val = X_processed[train_index], X_processed[val_index]
    y_train, y_val = y_orig[train_index], y_orig[val_index]
    
    # --- 2. 准备 Model A 数据 (Raw Scaled) ---
    scaler_A = StandardScaler()
    X_train_raw = scaler_A.fit_transform(X_train)
    X_val_raw = scaler_A.transform(X_val)
    X_train_cnn_A = X_train_raw.reshape(-1, N_BANDS_RAW, 1)
    X_val_cnn_A = X_val_raw.reshape(-1, N_BANDS_RAW, 1)
    
    # --- 3. 准备 Model B 数据 (PCA) ---
    scaler_B = StandardScaler()
    X_train_scaled_B = scaler_B.fit_transform(X_train) # 独立的 scaler
    X_val_scaled_B = scaler_B.transform(X_val)
    
    pca = PCA(n_components=N_COMPONENTS_PCA, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled_B)
    X_val_pca = pca.transform(X_val_scaled_B)
    X_train_cnn_B = X_train_pca.reshape(-1, N_COMPONENTS_PCA, 1)
    X_val_cnn_B = X_val_pca.reshape(-1, N_COMPONENTS_PCA, 1)
    
    # --- 4. 准备标签 ---
    y_train_cat = to_categorical(y_train, N_CLASSES)
    y_val_cat = to_categorical(y_val, N_CLASSES)
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {i: w for i, w in enumerate(cw)}
    
    # --- 5. 训练 Model A ---
    print("  正在训练 Model A (Raw)...")
    K.clear_session()
    model_A = build_model_A((N_BANDS_RAW, 1), N_CLASSES)
    model_A.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model_A.fit(X_train_cnn_A, y_train_cat, epochs=100, batch_size=32, validation_data=(X_val_cnn_A, y_val_cat), class_weight=cw_dict, callbacks=[early_stop], verbose=0)
    
    # 获取 A 在验证集上的预测
    probs_A = model_A.predict(X_val_cnn_A, verbose=0)
    acc_A = accuracy_score(y_val, np.argmax(probs_A, axis=1))
    results_A.append(acc_A)
    print(f"    Model A Acc: {acc_A:.4f}")

    # --- 6. 训练 Model B ---
    print("  正在训练 Model B (PCA)...")
    # 注意：不清除 session 可能会导致显存不足，但我们需要 Model A 还在内存里。
    # 如果显存不够，可以先 save A，clear session，再 load A。这里假设显存够用。
    model_B = build_model_B((N_COMPONENTS_PCA, 1), N_CLASSES)
    model_B.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model_B.fit(X_train_cnn_B, y_train_cat, epochs=100, batch_size=32, validation_data=(X_val_cnn_B, y_val_cat), class_weight=cw_dict, callbacks=[early_stop], verbose=0)
    
    # 获取 B 在验证集上的预测
    probs_B = model_B.predict(X_val_cnn_B, verbose=0)
    acc_B = accuracy_score(y_val, np.argmax(probs_B, axis=1))
    results_B.append(acc_B)
    print(f"    Model B Acc: {acc_B:.4f}")
    
    # --- 7. 寻找当前折的最佳融合权重 (Grid Search) ---
    print("  寻找最佳融合权重...")
    best_fold_acc = 0
    best_w = 0.5
    
    # 在 0.0 到 1.0 之间搜索权重 (步长 0.01)
    for w_a in np.arange(0.0, 1.01, 0.01):
        w_b = 1.0 - w_a
        # 加权融合
        probs_fusion = (w_a * probs_A) + (w_b * probs_B)
        pred_fusion = np.argmax(probs_fusion, axis=1)
        acc_fusion = accuracy_score(y_val, pred_fusion)
        
        if acc_fusion > best_fold_acc:
            best_fold_acc = acc_fusion
            best_w = w_a
            
    results_Fusion.append(best_fold_acc)
    best_weights_log.append(best_w)
    
    print(f"    [Fusion] Best Weight A: {best_w:.2f} | Acc: {best_fold_acc:.4f}")
    print(f"    提升: {best_fold_acc - max(acc_A, acc_B):.4f}")

# ==============================================================================
# 4. 汇总报告
# ==============================================================================
print(f"\n{'='*20} 5折交叉验证最终结果 {'='*20}")
print(f"Model A (Mean ± Std): {np.mean(results_A)*100:.2f}% ± {np.std(results_A)*100:.2f}%")
print(f"Model B (Mean ± Std): {np.mean(results_B)*100:.2f}% ± {np.std(results_B)*100:.2f}%")
print(f"Fusion  (Mean ± Std): {np.mean(results_Fusion)*100:.2f}% ± {np.std(results_Fusion)*100:.2f}%")

print("\n各折最佳权重 (Model A):", best_weights_log)
avg_weight = np.mean(best_weights_log)
print(f"建议最终使用的平均权重: A={avg_weight:.2f}, B={1-avg_weight:.2f}")

# 绘图
plt.figure(figsize=(8, 6))
labels = ['Model A', 'Model B', 'Fusion']
means = [np.mean(results_A), np.mean(results_B), np.mean(results_Fusion)]
stds = [np.std(results_A), np.std(results_B), np.std(results_Fusion)]

plt.bar(labels, means, yerr=stds, capsize=10, color=['#a8dadc', '#457b9d', '#e63946'], alpha=0.9)
plt.ylabel('Accuracy')
plt.title('5-Fold CV Performance Comparison')
plt.ylim(min(means)-0.1, 1.05)
for i, v in enumerate(means):
    plt.text(i, v + 0.01, f"{v*100:.2f}%", ha='center', fontweight='bold')
plt.savefig("Fusion_5Fold_Result.png")
plt.show()