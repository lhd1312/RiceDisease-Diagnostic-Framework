import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold  
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, recall_score, accuracy_score

# =========================================================
# VVVV --- 修改点 1: 更新导入，加入 CBAM 和 MSC 所需的层 --- VVVV
# =========================================================
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, 
    SeparableConv1D, Input, Add, Activation, 
    LayerNormalization, 
    # 新增的层用于 CBAM 和 多尺度
    GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, Multiply, Concatenate
)
# ^^^^ --------------------------------------------------- ^^^^

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import joblib 

# ... [设置字体、随机种子、常量和路径的代码保持不变] ...
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. 定义常量和路径 ---
DATA_DIR = Path('DATA'); VNIR_FILE = DATA_DIR / 'VNIR.xlsx'; SWIR_FILE = DATA_DIR / 'SWIR.xlsx'
SHEET_NAMES = [
    'CK', 'DWB_01_Resistant', 'DWB_02_Resistant', 'DWB_03_Resistant', 'DWB_04_Resistant',
    'DWB_05_Susceptible', 'BYK_01_Resistant', 'BYK_02_Resistant', 'BYK_03_Resistant', 'BYK_04_Susceptible'
]
label_map = {name: i for i, name in enumerate(SHEET_NAMES)}
N_CLASSES = 10

# ... [第 2 节 - 数据加载，保持不变] ...
features = []; labels = []
all_bands = np.array([])
print("开始加载和合并数据...")
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
    N_BANDS_RAW = X_orig.shape[1]
    print(f"\n--- 数据加载完成 (原始 {N_BANDS_RAW} bands) ---")
except Exception as e: 
    print(f"加载数据时发生错误: {e}"); exit()

# ... [第 2.5 节 - 光谱裁剪，保持不变] ...
print(f"\n--- 2.5. 正在裁剪光谱 ---")
WAVELENGTH_RANGES_TO_KEEP = [
    (420, 900),    # (请修改)
    (1000, 1700)   # (请修改)
]
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
print("开始光谱预处理 (SG+SNV)...")
X_processed = savgol_filter(X_orig, window_length=11, polyorder=3, axis=1)
mean_row = np.mean(X_processed, axis=1, keepdims=True)
std_row = np.std(X_processed, axis=1, keepdims=True)
X_processed = (X_processed - mean_row) / (std_row + 1e-8)

X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y_orig, test_size=0.2, random_state=42, stratify=y_orig
)
print(f"训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")

scaler_raw = StandardScaler()
X_train_raw = scaler_raw.fit_transform(X_train)
X_val_raw = scaler_raw.transform(X_val)
X_train_raw_cnn = X_train_raw.reshape(X_train_raw.shape[0], N_BANDS_RAW, 1)
X_val_raw_cnn = X_val_raw.reshape(X_val_raw.shape[0], N_BANDS_RAW, 1)
joblib.dump(scaler_raw, 'scaler_MSC_CBAM.pkl') 

y_train_cnn = to_categorical(y_train, N_CLASSES)
y_val_cnn = to_categorical(y_val, N_CLASSES)
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}

# ==============================================================================
# VVVV --- 修改点 2: 定义 CBAM 和 Multi-Scale 函数 --- VVVV
# ==============================================================================

def cbam_block(input_feature, ratio=8):
    """1D CBAM: 通道注意力 + 空间注意力"""
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
    """多尺度卷积模块"""
    b1 = SeparableConv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    b2 = SeparableConv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    b3 = SeparableConv1D(32, kernel_size=9, padding='same', activation='relu')(inputs)
    x = Concatenate()([b1, b2, b3])
    x = BatchNormalization()(x)
    return x

def build_model(input_shape, n_classes):
    """构建模型函数，方便在循环中多次调用"""
    inputs = Input(shape=input_shape)
    
    # 1. MSC
    x = multiscale_block(inputs) 
    x = MaxPooling1D(pool_size=2)(x)
    
    # 2. 融合
    x = SeparableConv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # 3. CBAM
    x = cbam_block(x, ratio=8)
    x = MaxPooling1D(pool_size=2)(x) 
    
    # 4. Head
    x = Flatten()(x)
    x = Dense(192, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ==============================================================================
# 2. 执行 5折交叉验证 (5-Fold Cross Validation)
# ==============================================================================
print(f"\n========== 开始 5折交叉验证 ==========")

# 初始化 KFold
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# 用于存储结果
fold_accuracies = []
fold_recalls = []
confusion_matrices = []

# 全局最佳模型保存逻辑 (可选)
best_global_acc = 0.0

# 开始循环
for fold_no, (train_index, val_index) in enumerate(skf.split(X_processed, y_orig)):
    print(f"\n>>> 正在处理第 {fold_no + 1} / {k_folds} 折...")
    
    # 1. 划分数据
    X_train_fold, X_val_fold = X_processed[train_index], X_processed[val_index]
    y_train_fold, y_val_fold = y_orig[train_index], y_orig[val_index]
    
    # 2. 数据标准化 (StandardScaler) 
    # 注意：必须fit在当前折的训练集上，然后transform验证集，防止数据泄露
    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train_fold)
    X_val_scaled = scaler_fold.transform(X_val_fold)
    
    # 3. Reshape 为 CNN 输入 (Samples, Bands, 1)
    N_BANDS_RAW = X_train_scaled.shape[1]
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], N_BANDS_RAW, 1)
    X_val_cnn = X_val_scaled.reshape(X_val_scaled.shape[0], N_BANDS_RAW, 1)
    
    # 4. 标签 One-hot
    y_train_cat = to_categorical(y_train_fold, N_CLASSES)
    y_val_cat = to_categorical(y_val_fold, N_CLASSES)
    
    # 5. 计算类别权重
    class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}
    
    # 6. 清理旧图并构建新模型
    K.clear_session() # 清除显存中的旧模型
    model = build_model((N_BANDS_RAW, 1), N_CLASSES)
    
    # 7. 训练
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    history = model.fit(
        X_train_cnn, y_train_cat,
        epochs=150, # 可以根据需要调整
        batch_size=32,
        validation_data=(X_val_cnn, y_val_cat),
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=0 # 设为0或1，避免每个epoch都刷屏，保持输出整洁
    )
    
    # 8. 评估当前折
    y_pred_prob = model.predict(X_val_cnn, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    acc = accuracy_score(y_val_fold, y_pred)
    fold_accuracies.append(acc)
    
    recall = recall_score(y_val_fold, y_pred, average='macro') # 或者 None 看每一类
    fold_recalls.append(recall)
    
    cm = confusion_matrix(y_val_fold, y_pred)
    confusion_matrices.append(cm)
    
    print(f"    第 {fold_no + 1} 折验证集准确率 (Accuracy): {acc:.4f}")
    
    # 保存这一折最好的模型 (可选，如果当前折比之前所有的都好)
    if acc > best_global_acc:
        best_global_acc = acc
        model.save("model_MSC_CBAM_best_fold.h5")
        joblib.dump(scaler_fold, 'scaler_best_fold.pkl') # 同时保存对应的scaler
        print(f"    [新纪录] 模型已保存，准确率: {acc:.4f}")

# ==============================================================================
# 3. 汇总输出结果
# ==============================================================================
print("\n" + "="*40)
print(f"5折交叉验证完成!")
print("="*40)

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print(f"各折准确率: {[f'{x:.4f}' for x in fold_accuracies]}")
print(f"平均准确率 (Mean Accuracy): {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")
print(f"平均召回率 (Mean Macro Recall): {np.mean(fold_recalls):.4f}")

# 汇总混淆矩阵 (将5次的结果加在一起，看总体分布)
total_cm = np.sum(confusion_matrices, axis=0)
print("\n累计混淆矩阵 (Sum of Confusion Matrices):")
print(total_cm)

# 打印最终的分类报告 (基于最后一折的，仅供参考，严谨的报告应基于所有折的预测汇总)
print("\n最后一折的 Classification Report:")
print(classification_report(y_val_fold, y_pred, digits=4))

print(f"\n最佳单折模型已保存为: model_MSC_CBAM_best_fold.h5 (Acc: {best_global_acc:.4f})")