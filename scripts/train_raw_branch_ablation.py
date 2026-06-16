import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import joblib 
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, 
    SeparableConv1D, Input, Add, Activation, 
    GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, Multiply, Concatenate
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# 保持之前的环境设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. 模块定义: CBAM 和 Multi-Scale Block
# ==============================================================================

def cbam_block(input_feature, ratio=8):
    """
    1D CBAM: 通道注意力 + 空间注意力
    """
    # --- Channel Attention (CAM) ---
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
    
    # --- Spatial Attention (SAM) ---
    avg_pool = tf.reduce_mean(x_refined_c, axis=channel_axis, keepdims=True)
    max_pool = tf.reduce_max(x_refined_c, axis=channel_axis, keepdims=True)
    concat = Concatenate(axis=channel_axis)([avg_pool, max_pool])
    
    cbam_feature = Conv1D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    x_refined_s = Multiply()([x_refined_c, cbam_feature])
    
    return x_refined_s

def multiscale_block(inputs):
    """
    多尺度卷积模块: 并行使用 3, 5, 9 卷积核
    """
    # 分支1: 小感受野 (细节)
    b1 = SeparableConv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    # 分支2: 中感受野
    b2 = SeparableConv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    # 分支3: 大感受野 (趋势)
    b3 = SeparableConv1D(32, kernel_size=9, padding='same', activation='relu')(inputs)
    
    # 拼接特征
    x = Concatenate()([b1, b2, b3])
    x = BatchNormalization()(x)
    return x

# ==============================================================================
# 2. 构建递进式模型 (Factory Function)
# ==============================================================================

def build_progressive_model(mode, input_shape, n_classes):
    """
    mode: 
        'baseline'  -> 普通 1D CNN
        'msc'       -> 1D CNN + Multi-Scale
        'msc_cbam'  -> 1D CNN + Multi-Scale + CBAM
    """
    inputs = Input(shape=input_shape)
    
    # --- 第一阶段：特征提取 ---
    if mode == 'baseline':
        # [Baseline] 普通卷积
        x = SeparableConv1D(32, kernel_size=9, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
    else:
        # [MSC & MSC+CBAM] 替换为多尺度模块
        # 这里替换了第一层，让模型一开始就看到不同尺度的特征
        x = multiscale_block(inputs)
    
    x = MaxPooling1D(pool_size=2)(x)
    
    # --- 第二阶段：进一步提取 (所有模型保持一致，或随通道数自适应) ---
    # 注意：MSC输出的通道数是 32*3=96，Baseline是 32。
    # 为了公平，第二层我们统一用 64 个卷积核进行特征融合/降维
    x = SeparableConv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # --- 第三阶段：注意力机制 (仅 msc_cbam) ---
    if mode == 'msc_cbam':
        print(">>> 正在插入 CBAM 模块...")
        x = multiscale_block(inputs)
        x = Conv1D(filters=64, kernel_size=1, activation='relu')(x)
        x = cbam_block(x)
        
    # --- 第四阶段：分类头 ---
    x = Flatten()(x)
    x = Dense(192, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=f"Model_{mode}")
    return model

# ==============================================================================
# 3. 执行递进式消融实验
# ==============================================================================
# --- 1. 定义常量和路径 ---
DATA_DIR = Path('DATA'); VNIR_FILE = DATA_DIR / 'VNIR.xlsx'; SWIR_FILE = DATA_DIR / 'SWIR.xlsx'
SHEET_NAMES = [
    'CK', 'DWB_01_Resistant', 'DWB_02_Resistant', 'DWB_03_Resistant', 'DWB_04_Resistant',
    'DWB_05_Susceptible', 'BYK_01_Resistant', 'BYK_02_Resistant', 'BYK_03_Resistant', 'BYK_04_Susceptible'
]
label_map = {name: i for i, name in enumerate(SHEET_NAMES)}
N_CLASSES = 10

# ... [第 2 节 - 数据加载，保持不变] ...
# --- 2. 数据加载 ---
features = []; labels = []
all_bands = np.array([])
print("开始加载和合并数据 (Model A)...")
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
    if len(all_bands) != X_orig.shape[1]:
        print(f"!!! 警告: 波长数 ({len(all_bands)}) 与数据列数 ({X_orig.shape[1]}) 不匹配!")
        N_BANDS_RAW = X_orig.shape[1]
    else:
        N_BANDS_RAW = X_orig.shape[1] 
        print(f"\n--- 数据加载完成 (原始 {N_BANDS_RAW} bands) ---")
except Exception as e: 
    print(f"加载数据时发生错误: {e}"); exit()


# ... [第 2.5 节 - 光谱裁剪，保持不变] ...
# ==========================================================
# --- 2.5. 光谱裁剪 (Trimming) ---
# ==========================================================
print(f"\n--- 2.5. 正在裁剪光谱 ---")
WAVELENGTH_RANGES_TO_KEEP = [
    (420, 900),    # (请修改)
    (1000, 1700)   # (请修改)
]
print(f"裁剪前 X_orig 形状: {X_orig.shape}")
try:
    idx_keep = np.zeros(all_bands.shape, dtype=bool)
    for (start, end) in WAVELENGTH_RANGES_TO_KEEP:
        idx_keep = idx_keep | ((all_bands >= start) & (all_bands <= end))
except Exception as e:
    print(f"!!! 错误: 创建裁剪索引时失败: {e}"); exit()
X_orig = X_orig[:, idx_keep]
N_BANDS_RAW = X_orig.shape[1] 
if N_BANDS_RAW == 0:
    print("!!! 错误: 裁剪后没有剩下任何波段! !!!"); exit()
print(f"裁剪后 X_orig 形状: {X_orig.shape}")
print(f"将使用 {N_BANDS_RAW} 个波段进行训练。")


# ... [第 3, 4, 5 节 - 预处理, 划分, 准备数据，保持不变] ...
# --- 3. 光谱预处理 (SG+SNV) ---
print("开始光谱预处理 (SG+SNV)...")
X_processed = savgol_filter(X_orig, window_length=11, polyorder=3, axis=1)
mean_row = np.mean(X_processed, axis=1, keepdims=True)
std_row = np.std(X_processed, axis=1, keepdims=True)
X_processed = (X_processed - mean_row) / (std_row + 1e-8)
# --- 4. 划分训练集和验证集 ---
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y_orig, test_size=0.2, random_state=42, stratify=y_orig
)
print(f"训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")
# --- 5. 准备 "Raw" 输入 ---
print(f"准备 Raw 输入 ( {N_BANDS_RAW} 波段)...")
scaler_raw = StandardScaler()
X_train_raw = scaler_raw.fit_transform(X_train)
X_val_raw = scaler_raw.transform(X_val)
X_train_raw_cnn = X_train_raw.reshape(X_train_raw.shape[0], N_BANDS_RAW, 1)
X_val_raw_cnn = X_val_raw.reshape(X_val_raw.shape[0], N_BANDS_RAW, 1)
joblib.dump(scaler_raw, 'scaler_A_trimmed.pkl') 
print("scaler_A_trimmed.pkl 已保存。")
y_train_cnn = to_categorical(y_train, N_CLASSES)
y_val_cnn = to_categorical(y_val, N_CLASSES)
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}



modes = ['baseline', 'msc', 'msc_cbam']
results = {}

print(f"\n{'='*20} 开始递进式消融实验 {'='*20}")
print(f"实验顺序: {modes}")

for mode in modes:
    print(f"\n\n>>> 正在训练模型: [ {mode.upper()} ]")
    
    # 1. 清理与构建
    tf.keras.backend.clear_session()
    model = build_progressive_model(mode, input_shape=(N_BANDS_RAW, 1), n_classes=N_CLASSES)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # 2. 训练
    early_stop = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)
    
    history = model.fit(
        X_train_raw_cnn, y_train_cnn,
        epochs=200,  # 建议设为 200
        batch_size=32,
        validation_data=(X_val_raw_cnn, y_val_cnn),
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=0 # 静默训练，只打印结果
    )
    
    # 3. 评估指标
    val_loss, val_acc = model.evaluate(X_val_raw_cnn, y_val_cnn, verbose=0)
    y_pred = np.argmax(model.predict(X_val_raw_cnn, verbose=0), axis=1)
    y_true = np.argmax(y_val_cnn, axis=1)
    
    report = classification_report(y_true, y_pred, output_dict=True)
    macro_f1 = report['macro avg']['f1-score']
    
    print(f"[{mode.upper()}] 完成 -> Val Acc: {val_acc:.4f} | Macro F1: {macro_f1:.4f}")
    
    results[mode] = {
        'history': history.history,
        'val_acc': val_acc,
        'macro_f1': macro_f1,
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    model.save(f"Ablation_{mode}.h5")

# ==============================================================================
# 4. 结果可视化 (证明有效性)
# ==============================================================================

print(f"\n{'='*20} 最终结果对比 {'='*20}")
print(f"{'Model':<15} | {'Acc':<8} | {'F1-Score':<8} | {'Improvement (vs Base)'}")
print("-" * 60)

base_acc = results['baseline']['val_acc']

for mode in modes:
    acc = results[mode]['val_acc']
    f1 = results[mode]['macro_f1']
    diff = acc - base_acc
    diff_str = f"+{diff:.2%}" if diff > 0 else f"{diff:.2%}"
    if mode == 'baseline': diff_str = "-"
    
    print(f"{mode:<15} | {acc:.4f}   | {f1:.4f}     | {diff_str}")

# 绘图
plt.figure(figsize=(14, 6))

# 准确率曲线
plt.subplot(1, 2, 1)
colors = ['blue', 'orange', 'red']
for i, mode in enumerate(modes):
    plt.plot(results[mode]['history']['val_accuracy'], 
             label=f"{mode} (Best: {results[mode]['val_acc']:.3f})", 
             color=colors[i], linewidth=2)
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# Loss 曲线
plt.subplot(1, 2, 2)
for i, mode in enumerate(modes):
    plt.plot(results[mode]['history']['val_loss'], 
             label=f"{mode}", 
             color=colors[i], linewidth=2)
plt.title('Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Progressive_Ablation_Result.png')
plt.show()