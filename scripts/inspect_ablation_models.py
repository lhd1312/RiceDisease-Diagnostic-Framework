import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score
from tensorflow.keras.models import load_model

# ==========================================
# 0. 请在这里填入你找到的 .h5 文件路径！
# ==========================================
# 如果你只测试 PM，就把 RMC 相关的注释掉
MODEL_PATHS = {
    # PM 分支模型
    'PM 1': 'Model_B_baseline.h5',  # 替换为实际路径
    'PM 2': 'Model_B_msc.h5',       # 替换为实际路径
    'PM 3': 'Model_B_msc_cbam.h5',  # 替换为实际路径
    
    # 如果你也想顺便跑 RMC，取消下面注释并填入路径
    # 'RMC 1': 'Ablation_RMC_baseline.h5',
    # 'RMC 2': 'Ablation_RMC_msc.h5',
    # 'RMC 3': 'Ablation_RMC_msc_cbam.h5',
}

# ==========================================
# 1. 重建与之前绝对一致的测试集 (随机种子必须是42)
# ==========================================
DATA_DIR = Path('DATA')
VNIR_FILE = DATA_DIR / 'VNIR.xlsx'
SWIR_FILE = DATA_DIR / 'SWIR.xlsx'
SHEET_NAMES = [
    'CK', 'DWB_01_Resistant', 'DWB_02_Resistant', 'DWB_03_Resistant', 'DWB_04_Resistant',
    'DWB_05_Susceptible', 'BYK_01_Resistant', 'BYK_02_Resistant', 'BYK_03_Resistant', 'BYK_04_Susceptible'
]
label_map = {name: i for i, name in enumerate(SHEET_NAMES)}

# 加载数据
features = []; labels = []
all_bands = np.array([])
print("正在加载数据以重建一致的测试集...")
with pd.ExcelFile(VNIR_FILE) as vnir_xls, pd.ExcelFile(SWIR_FILE) as swir_xls:
    for sheet in SHEET_NAMES:
        vnir_data = pd.read_excel(vnir_xls, sheet_name=sheet, header=0)
        swir_data = pd.read_excel(swir_xls, sheet_name=sheet, header=0)
        if not all_bands.any():
            all_bands = np.concatenate([vnir_data.columns.astype(float), swir_data.columns.astype(float)])
        features.append(pd.concat([vnir_data, swir_data], axis=1))
        labels.append(np.full(features[-1].shape[0], label_map[sheet]))
        
X_orig = pd.concat(features, ignore_index=True).values
y_orig = np.concatenate(labels)

# 裁剪并预处理 (SG+SNV)
idx_keep = ((all_bands >= 420) & (all_bands <= 900)) | ((all_bands >= 1000) & (all_bands <= 1700))
X_orig = X_orig[:, idx_keep]

X_processed = savgol_filter(X_orig, window_length=11, polyorder=3, axis=1)
mean_row = np.mean(X_processed, axis=1, keepdims=True)
std_row = np.std(X_processed, axis=1, keepdims=True)
X_processed = (X_processed - mean_row) / (std_row + 1e-8)

# 严格使用原来的随机种子 42 划分，确保测试集与论文原版完全一致
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y_orig, test_size=0.2, random_state=42, stratify=y_orig
)

# 准备 RMC 格式数据 (Raw)
scaler_raw = StandardScaler()
X_train_raw = scaler_raw.fit_transform(X_train)
X_val_raw = scaler_raw.transform(X_val)
X_val_rmc_cnn = X_val_raw.reshape(X_val_raw.shape[0], X_val_raw.shape[1], 1)

# 准备 PM 格式数据 (PCA)
scaler_pm = StandardScaler()
X_train_scaled = scaler_pm.fit_transform(X_train)
X_val_scaled = scaler_pm.transform(X_val)
pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_val_pm_cnn = X_val_pca.reshape(X_val_pca.shape[0], 100, 1)

# ==========================================
# 2. 读取模型并计算指标
# ==========================================
print(f"\n{'='*20} 原始 .h5 模型评估结果 {'='*20}")
print(f"{'Model Name':<12} | {'OA(Acc)':<8} | {'F1-Score':<8} | {'Kappa':<8} | {'RB-OA':<8} | {'BLB-OA':<8}")
print("-" * 75)

for model_name, path in MODEL_PATHS.items():
    if not Path(path).exists():
        print(f"{model_name:<12} | 文件未找到: {path}")
        continue
        
    try:
        # compile=False 极其重要！它可以避免加载自定义损失或优化器导致的报错
        model = load_model(path, compile=False)
        
        # 自动判断该用 Raw 数据还是 PCA 数据
        input_shape = model.input_shape[-2]
        if input_shape == 100:
            X_test_input = X_val_pm_cnn
        else:
            X_test_input = X_val_rmc_cnn
            
        # 预测
        y_pred_probs = model.predict(X_test_input, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # 计算全局指标
        oa = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        macro_f1 = report['macro avg']['f1-score']
        kappa = cohen_kappa_score(y_val, y_pred)
        
        # 计算特定疾病大类准确率
        rb_mask = np.isin(y_val, [1, 2, 3, 4, 5])
        blb_mask = np.isin(y_val, [6, 7, 8, 9])
        rb_oa = np.mean(y_val[rb_mask] == y_pred[rb_mask]) if np.sum(rb_mask) > 0 else 0
        blb_oa = np.mean(y_val[blb_mask] == y_pred[blb_mask]) if np.sum(blb_mask) > 0 else 0
        
        # 打印符合要求的结果表格
        print(f"{model_name:<12} | {oa:.4f}   | {macro_f1:.4f}   | {kappa:.4f}   | {rb_oa:.4f}   | {blb_oa:.4f}")
        
    except Exception as e:
        print(f"{model_name:<12} | 运行出错: {e}")

print("-" * 75)
print("直接将上面的结果填入你的 Tab. 6 即可。祝投 PMS 顺利！")