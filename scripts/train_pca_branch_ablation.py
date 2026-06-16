import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, recall_score

# --- 核心修改: 使用 Functional API 以支持 MSC 和 CBAM ---
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, MaxPooling1D, Flatten, Dropout, BatchNormalization, 
    SeparableConv1D, Input, Add, Activation, Concatenate, Multiply,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, Conv1D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. 定义 CBAM 和 MSC 模块
# ==============================================================================

def cbam_block(input_feature, ratio=8):
    """ 1D CBAM 模块 """
    # --- Channel Attention ---
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
    
    # --- Spatial Attention ---
    avg_pool = tf.reduce_mean(x_refined_c, axis=channel_axis, keepdims=True)
    max_pool = tf.reduce_max(x_refined_c, axis=channel_axis, keepdims=True)
    concat = Concatenate(axis=channel_axis)([avg_pool, max_pool])
    
    cbam_feature = Conv1D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    x_refined_s = Multiply()([x_refined_c, cbam_feature])
    
    return x_refined_s

def multiscale_block(inputs):
    """ 多尺度卷积模块 (针对 PCA 只有 100 个特征，使用稍小的卷积核) """
    # 分支1: 细节
    b1 = SeparableConv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    # 分支2: 中等
    b2 = SeparableConv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    # 分支3: 宏观
    b3 = SeparableConv1D(32, kernel_size=7, padding='same', activation='relu')(inputs)
    
    x = Concatenate()([b1, b2, b3])
    x = BatchNormalization()(x)
    return x

def build_model_B_variant(mode, input_shape, n_classes):
    """
    根据 mode 构建不同变体的 Model B (PCA-CNN)
    mode: 'baseline', 'msc', 'cbam', 'msc_cbam'
    """
    inputs = Input(shape=input_shape)
    
    # --- 第一层卷积 ---
    if 'msc' in mode:
        # 如果是多尺度，使用 MSC Block 替换第一层
        x = multiscale_block(inputs) # 输出通道 32*3=96
    else:
        # 否则使用原始定义的 Simple Conv
        x = SeparableConv1D(filters=32, kernel_size=7, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
    
    x = MaxPooling1D(pool_size=2)(x)

    # --- 第二层卷积 ---
    # 为了保持公平，如果上一层是 MSC (96通道)，这一层将其降维或保持特征
    # 原模型第二层 filters=96
    x = SeparableConv1D(filters=96, kernel_size=7, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # --- CBAM 插入点 ---
    if 'cbam' in mode:
        print(f"[{mode}] Adding CBAM...")
        # 如果使用了 MSC，建议先用 1x1 卷积融合一下再加 CBAM (方案二)
        if 'msc' in mode:
             x = Conv1D(filters=96, kernel_size=1, activation='relu')(x)
        x = cbam_block(x)

    # --- 分类头 (保持原 Model B 参数) ---
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name=f"Model_B_{mode}")

# ==============================================================================
# 2. 数据处理 (保持不变)
# ==============================================================================

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
print("开始加载和合并数据 (Model B Experimental)...")
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
    print(f"加载数据错误: {e}"); exit()

# 裁剪
WAVELENGTH_RANGES_TO_KEEP = [(420, 900), (1000, 1700)]
idx_keep = np.zeros(all_bands.shape, dtype=bool)
for (start, end) in WAVELENGTH_RANGES_TO_KEEP:
    idx_keep = idx_keep | ((all_bands >= start) & (all_bands <= end))
X_orig = X_orig[:, idx_keep]

# 预处理
print("SG + SNV + PCA 处理中...")
X_processed = savgol_filter(X_orig, window_length=11, polyorder=3, axis=1)
mean_row = np.mean(X_processed, axis=1, keepdims=True)
std_row = np.std(X_processed, axis=1, keepdims=True)
X_processed = (X_processed - mean_row) / (std_row + 1e-8)

X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y_orig, test_size=0.2, random_state=42, stratify=y_orig
)

# PCA
scaler_pca = StandardScaler()
X_train_scaled = scaler_pca.fit_transform(X_train)
X_val_scaled = scaler_pca.transform(X_val)
pca = PCA(n_components=N_COMPONENTS_PCA, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# Reshape for CNN
X_train_pca_cnn = X_train_pca.reshape(X_train_pca.shape[0], N_COMPONENTS_PCA, 1)
X_val_pca_cnn = X_val_pca.reshape(X_val_pca.shape[0], N_COMPONENTS_PCA, 1)

joblib.dump(scaler_pca, 'scaler_B_trimmed.pkl')
joblib.dump(pca, 'pca_B_trimmed.pkl')

y_train_cnn = to_categorical(y_train, N_CLASSES)
y_val_cnn = to_categorical(y_val, N_CLASSES)
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}

# ==============================================================================
# 3. 实验循环 (Baseline, MSC, CBAM, MSC+CBAM)
# ==============================================================================

experiment_modes = ['baseline', 'msc', 'cbam', 'msc_cbam']
results = {}

print(f"\n{'='*30}")
print(f"开始 Model B (PCA-CNN) 消融实验: {experiment_modes}")
print(f"{'='*30}")

for mode in experiment_modes:
    print(f"\n>>> 正在训练 Model B 变体: [ {mode.upper()} ]")
    
    tf.keras.backend.clear_session()
    
    model = build_model_B_variant(mode, input_shape=(N_COMPONENTS_PCA, 1), n_classes=N_CLASSES)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
    
    history = model.fit(
        X_train_pca_cnn, y_train_cnn,
        epochs=200, 
        batch_size=32,
        validation_data=(X_val_pca_cnn, y_val_cnn),
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=0 # 保持清爽，只打印结果
    )
    
    # 评估
    val_loss, val_acc = model.evaluate(X_val_pca_cnn, y_val_cnn, verbose=0)
    y_pred = np.argmax(model.predict(X_val_pca_cnn, verbose=0), axis=1)
    
    report = classification_report(y_val, y_pred, output_dict=True)
    macro_f1 = report['macro avg']['f1-score']
    
    print(f"[{mode.upper()}] 结果 -> Val Acc: {val_acc:.4f} | Macro F1: {macro_f1:.4f}")
    
    results[mode] = {
        'val_acc': val_acc,
        'macro_f1': macro_f1,
        'history': history.history
    }
    
    model.save(f"Model_B_{mode}.h5")

# ==============================================================================
# 4. 结果对比
# ==============================================================================
print(f"\n{'='*20} Model B (PCA) 实验汇总 {'='*20}")
print(f"{'Mode':<15} | {'Acc':<8} | {'F1-Score':<8}")
print("-" * 40)
for mode in experiment_modes:
    print(f"{mode:<15} | {results[mode]['val_acc']:.4f}   | {results[mode]['macro_f1']:.4f}")

# 绘图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for mode in experiment_modes:
    plt.plot(results[mode]['history']['val_accuracy'], label=mode)
plt.title('Validation Accuracy (PCA-CNN)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for mode in experiment_modes:
    plt.plot(results[mode]['history']['val_loss'], label=mode)
plt.title('Validation Loss (PCA-CNN)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Model_B_Ablation_Results.png")
plt.show()