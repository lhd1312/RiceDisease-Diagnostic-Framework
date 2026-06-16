import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold  # <--- 核心修改
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, MaxPooling1D, Flatten, Dropout, BatchNormalization, 
    SeparableConv1D, Input, Add, Activation, Concatenate, Multiply,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, Conv1D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K # 用于清理内存

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. 定义模块 (保持不变)
# ==============================================================================

def cbam_block(input_feature, ratio=8):
    """ 1D CBAM 模块 """
    channel_axis = -1
    filters = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(filters, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling1D()(input_feature)    
    avg_pool = Reshape((1, filters))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = Reshape((1, filters))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    x_refined_c = Multiply()([input_feature, cbam_feature])
    
    avg_pool = tf.reduce_mean(x_refined_c, axis=channel_axis, keepdims=True)
    max_pool = tf.reduce_max(x_refined_c, axis=channel_axis, keepdims=True)
    concat = Concatenate(axis=channel_axis)([avg_pool, max_pool])
    
    cbam_feature = Conv1D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    x_refined_s = Multiply()([x_refined_c, cbam_feature])
    
    return x_refined_s

def multiscale_block(inputs):
    """ 多尺度卷积模块 """
    b1 = SeparableConv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    b2 = SeparableConv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    b3 = SeparableConv1D(32, kernel_size=7, padding='same', activation='relu')(inputs)
    
    x = Concatenate()([b1, b2, b3])
    x = BatchNormalization()(x)
    return x

def build_model_B_variant(mode, input_shape, n_classes):
    """
    根据 mode 构建不同变体的 Model B (PCA-CNN)
    """
    inputs = Input(shape=input_shape)
    
    # --- 第一层 ---
    if 'msc' in mode:
        x = multiscale_block(inputs) 
    else:
        x = SeparableConv1D(filters=32, kernel_size=7, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
    
    x = MaxPooling1D(pool_size=2)(x)

    # --- 第二层 ---
    x = SeparableConv1D(filters=96, kernel_size=7, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # --- CBAM ---
    if 'cbam' in mode:
        # 如果是 MSC 后的 CBAM，通常通道数较多，可以直接加或者先降维
        if 'msc' in mode:
             x = Conv1D(filters=96, kernel_size=1, activation='relu')(x)
        x = cbam_block(x)

    # --- Head ---
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name=f"Model_B_{mode}")

# ==============================================================================
# 2. 数据加载与预处理 (SG + SNV)
# ==============================================================================
# ... [这部分保持不变，直到 savgol_filter 结束] ...
DATA_DIR = Path('DATA'); VNIR_FILE = DATA_DIR / 'VNIR.xlsx'; SWIR_FILE = DATA_DIR / 'SWIR.xlsx'
SHEET_NAMES = [
    'CK', 'DWB_01_Resistant', 'DWB_02_Resistant', 'DWB_03_Resistant', 'DWB_04_Resistant',
    'DWB_05_Susceptible', 'BYK_01_Resistant', 'BYK_02_Resistant', 'BYK_03_Resistant', 'BYK_04_Susceptible'
]
label_map = {name: i for i, name in enumerate(SHEET_NAMES)}
N_CLASSES = 10
N_COMPONENTS_PCA = 100 

features = []; labels = []
all_bands = np.array([])
print("开始加载数据...")
try:
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
except Exception as e: 
    print(f"数据加载失败: {e}"); exit()

WAVELENGTH_RANGES_TO_KEEP = [(420, 900), (1000, 1700)]
idx_keep = np.zeros(all_bands.shape, dtype=bool)
for (start, end) in WAVELENGTH_RANGES_TO_KEEP:
    idx_keep = idx_keep | ((all_bands >= start) & (all_bands <= end))
X_orig = X_orig[:, idx_keep]

print("正在进行 SG滤波 + SNV 处理...")
X_processed = savgol_filter(X_orig, window_length=11, polyorder=3, axis=1)
mean_row = np.mean(X_processed, axis=1, keepdims=True)
std_row = np.std(X_processed, axis=1, keepdims=True)
X_processed = (X_processed - mean_row) / (std_row + 1e-8)

# 注意：这里不再做 train_test_split，也不做全局 PCA
# 而是直接进入 5折交叉验证循环

# ==============================================================================
# 3. 5折交叉验证循环 (针对每种 Mode)
# ==============================================================================

experiment_modes = ['baseline', 'msc', 'cbam', 'msc_cbam']
final_results = {} # 存储最终的 Mean/Std

k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

print(f"\n{'='*40}")
print(f"开始 5折交叉验证消融实验")
print(f"Modes: {experiment_modes}")
print(f"{'='*40}")

for mode in experiment_modes:
    print(f"\n>>> 正在评估模式: [ {mode.upper()} ]")
    
    fold_accuracies = []
    best_fold_acc = 0.0
    
    # 开始 5折循环
    for fold_no, (train_index, val_index) in enumerate(skf.split(X_processed, y_orig)):
        
        # --- A. 数据划分 ---
        X_train_fold, X_val_fold = X_processed[train_index], X_processed[val_index]
        y_train_fold, y_val_fold = y_orig[train_index], y_orig[val_index]
        
        # --- B. 数据标准化 (StandardScaler) ---
        # 必须在 Fold 内部做 fit
        scaler_fold = StandardScaler()
        X_train_scaled = scaler_fold.fit_transform(X_train_fold)
        X_val_scaled = scaler_fold.transform(X_val_fold)
        
        # --- C. PCA 降维 ---
        # 必须在 Fold 内部做 fit，防止数据泄露
        pca_fold = PCA(n_components=N_COMPONENTS_PCA, random_state=42)
        X_train_pca = pca_fold.fit_transform(X_train_scaled)
        X_val_pca = pca_fold.transform(X_val_scaled)
        
        # --- D. Reshape 为 CNN 格式 (Batch, 100, 1) ---
        X_train_cnn = X_train_pca.reshape(X_train_pca.shape[0], N_COMPONENTS_PCA, 1)
        X_val_cnn = X_val_pca.reshape(X_val_pca.shape[0], N_COMPONENTS_PCA, 1)
        
        # --- E. 标签处理 ---
        y_train_cat = to_categorical(y_train_fold, N_CLASSES)
        y_val_cat = to_categorical(y_val_fold, N_CLASSES)
        class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
        class_weights_dict = {i: w for i, w in enumerate(class_weights_array)}
        
        # --- F. 构建并训练模型 ---
        K.clear_session() # 清理旧模型显存
        
        model = build_model_B_variant(mode, input_shape=(N_COMPONENTS_PCA, 1), n_classes=N_CLASSES)
        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        
        model.fit(
            X_train_cnn, y_train_cat,
            epochs=150, 
            batch_size=32,
            validation_data=(X_val_cnn, y_val_cat),
            class_weight=class_weights_dict,
            callbacks=[early_stop],
            verbose=0 # 静默训练
        )
        
        # --- G. 评估 ---
        y_pred = np.argmax(model.predict(X_val_cnn, verbose=0), axis=1)
        acc = accuracy_score(y_val_fold, y_pred)
        fold_accuracies.append(acc)
        
        print(f"    Fold {fold_no+1} Acc: {acc:.4f}")
        
        # 保存该模式下最好的那一折模型
        if acc > best_fold_acc:
            best_fold_acc = acc
            model.save(f"Best_Model_B_{mode}.h5")
            # 实际上这里应该保存对应的 scaler 和 pca，但在消融实验中通常省略，仅作最终模型部署时保存
            
    # --- 计算该模式的平均性能 ---
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    final_results[mode] = (mean_acc, std_acc)
    print(f" -> [ {mode.upper()} ] 平均准确率: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)")

# ==============================================================================
# 4. 最终结果汇总与可视化
# ==============================================================================
print(f"\n{'='*20} Model B (PCA-CNN) 5折交叉验证汇总 {'='*20}")
print(f"{'Mode':<15} | {'Mean Acc':<10} | {'Std Dev':<10}")
print("-" * 45)
for mode in experiment_modes:
    mean_v, std_v = final_results[mode]
    print(f"{mode:<15} | {mean_v:.4f}     | {std_v:.4f}")

# 绘制柱状图对比
modes = list(final_results.keys())
means = [final_results[m][0] for m in modes]
stds = [final_results[m][1] for m in modes]

plt.figure(figsize=(10, 6))
bars = plt.bar(modes, means, yerr=stds, capsize=10, color=['skyblue', 'lightgreen', 'orange', 'salmon'], alpha=0.8)
plt.title('Ablation Study: 5-Fold CV Accuracy (PCA-CNN)', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.05) # 稍微留点空间给 error bar
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱子上标数值
for bar, mean_val in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2, mean_val + 0.02, 
             f'{mean_val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("Model_B_5Fold_Results.png")
print("\n已保存结果图: Model_B_5Fold_Results.png")
plt.show()