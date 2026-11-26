"""Model constructors used in the revision experiments."""
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Concatenate, Conv1D, Dense, Dropout,
    Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D, Input, MaxPooling1D,
    Multiply, Reshape, SeparableConv1D,
)
from tensorflow.keras.models import Model


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


def rmc_multiscale_block(inputs):
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
        x = rmc_multiscale_block(inputs)

    x = MaxPooling1D(pool_size=2)(x)

    # --- 第二阶段：进一步提取 (所有模型保持一致，或随通道数自适应) ---
    # 注意：MSC输出的通道数是 32*3=96，Baseline是 32。
    # 为了公平，第二层我们统一用 64 个卷积核进行特征融合/降维
    x = SeparableConv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # --- 第三阶段：注意力机制 (仅 msc_cbam) ---
    if mode == 'msc_cbam':
        x = rmc_multiscale_block(inputs)
        x = Conv1D(filters=64, kernel_size=1, activation='relu')(x)
        x = cbam_block(x)

    # --- 第四阶段：分类头 ---
    x = Flatten()(x)
    x = Dense(192, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"Model_{mode}")
    return model


def pm_multiscale_block(inputs):
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
        x = pm_multiscale_block(inputs) # 输出通道 32*3=96
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


def replace_msc(pm_factory):
    def build(variant, shape, n_classes):
        original = pm_factory("msc", shape, n_classes)
        tail = original.layers[5:]
        assert [type(layer).__name__ for layer in tail] == [
            "BatchNormalization", "MaxPooling1D", "SeparableConv1D",
            "BatchNormalization", "MaxPooling1D", "Flatten", "Dense", "Dropout", "Dense"]
        # Keep the (100, 96) block output and the existing normalization and tail.
        inp = L.Input(shape)
        z = L.Dense(32, activation="relu")(L.Flatten()(inp))
        z = L.Dense(shape[0] * 96, activation="relu")(z)
        z = L.Reshape((shape[0], 96))(z)
        assert z.shape[1:] == tail[0].input.shape[1:]
        for layer in tail:
            z = type(layer).from_config(layer.get_config())(z)
        return tf.keras.Model(inp, z, name="PM_MLP_first_block")
    return build


def feature_concat(raw_factory, pca_factory):
    def build(variant, shapes, n_classes):
        raw = raw_factory("msc_cbam", shapes[0], n_classes)
        pm = pca_factory("msc", shapes[1], n_classes)
        features = L.Concatenate()([raw.layers[-2].output, pm.layers[-2].output])
        if variant == "mlp":
            features = L.Dense(64, activation="relu")(features)
            features = L.Dropout(.3)(features)
        prediction = L.Dense(n_classes, activation="softmax")(features)
        name = "FeatureConcatMLP" if variant == "mlp" else "FeatureConcat"
        return tf.keras.Model([raw.input, pm.input], prediction, name=name)
    return build
