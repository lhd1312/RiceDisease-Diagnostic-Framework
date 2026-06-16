import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, 
    Reshape, Multiply, Concatenate, Add, Activation, 
    SeparableConv1D, BatchNormalization, Lambda
)
import joblib
import numpy as np

# ==========================================================
# 1. 定义自定义模块 (必须与训练时完全一致)
# ==========================================================
def cbam_block(input_feature, ratio=8):
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
    b1 = SeparableConv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    b2 = SeparableConv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    b3 = SeparableConv1D(32, kernel_size=7, padding='same', activation='relu')(inputs)
    x = Concatenate()([b1, b2, b3])
    x = BatchNormalization()(x)
    return x

# ==========================================================
# 2. 加载模型和权重
# ==========================================================
print("正在加载子模型和最佳权重...")
custom_objects = {"cbam_block": cbam_block, "multiscale_block": multiscale_block}

try:
    # 加载子模型
    model_A = load_model("Ablation_msc_cbam.h5", custom_objects=custom_objects)
    model_B = load_model("Model_B_msc.h5", custom_objects=custom_objects)
    
    # 修改子模型名称，防止层名称冲突
    model_A._name = "Model_A_Branch"
    model_B._name = "Model_B_Branch"
    
    # 加载最佳权重 (如果没有这个文件，你可以手动指定 best_w_A = 0.6 这样)
    best_w_A = joblib.load('meta_model_weighted_A.pkl')
    print(f"加载到最佳权重 w_A: {best_w_A:.4f}")
    
except Exception as e:
    print(f"加载失败: {e}")
    # 如果没有权重文件，这里给一个默认值演示
    best_w_A = 0.5 
    print(f"使用默认权重 w_A: {best_w_A:.4f}")

# ==========================================================
# 3. 构建合并模型 (Stitching)
# ==========================================================
print("正在构建合并模型...")

# 1. 定义两个输入
# Model A 输入形状 (N_BANDS_RAW, 1)
input_A = Input(shape=model_A.input_shape[1:], name="Input_Raw_Spectra")
# Model B 输入形状 (N_COMPONENTS_PCA, 1)
input_B = Input(shape=model_B.input_shape[1:], name="Input_PCA_Spectra")

# 2. 获取子模型输出
out_A = model_A(input_A)
out_B = model_B(input_B)

# 3. 使用 Lambda 层应用加权平均 (确保可序列化)
# 公式: Final = w * A + (1-w) * B
def weighted_average(args):
    a, b = args
    w = best_w_A # 使用闭包捕获变量
    return (w * a) + ((1.0 - w) * b)

# 注意：为了让模型保存时记住 best_w_A，我们在 Lambda 中直接使用数值
# 或者使用 Add() 和 Multiply() 层组合
weight_layer_A = Lambda(lambda x: x * best_w_A, name="Weight_A")(out_A)
weight_layer_B = Lambda(lambda x: x * (1.0 - best_w_A), name="Weight_B")(out_B)

final_output = Add(name="Weighted_Sum")([weight_layer_A, weight_layer_B])

# 4. 实例化新模型
ensemble_model = Model(inputs=[input_A, input_B], outputs=final_output, name="Ensemble_Final")

# ==========================================================
# 4. 保存结果
# ==========================================================
ensemble_model.summary()

save_path = "Ensemble_Final.h5"
ensemble_model.save(save_path)
print(f"\n成功保存合并模型至: {save_path}")
print("提示: 推理时需要提供列表形式的输入: [X_raw, X_pca]")