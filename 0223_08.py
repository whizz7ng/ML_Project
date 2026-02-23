import tensorflow as tf
from tensorflow.keras import layers, models
import datetime
import numpy as np

# 1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. tf.data 파이프라인
# EfficientNetB3 최소 권장 입력: 96x96 이상 -> 32x32를 96x96으로 업스케일
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
IMG_SIZE = 64  # 32 -> 96으로 업스케일

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

def preprocess(x, y, training=False):
    # 96x96으로 리사이즈
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    # EfficientNet 전용 전처리 (0~1 -> -1~1 스케일 변환 내장)
    x = tf.keras.applications.efficientnet.preprocess_input(x * 255.0)
    if training:
        x = data_augmentation(x, training=True)
    # Label Smoothing을 위해 one-hot 변환 (CategoricalCrossentropy 사용)
    y = tf.one_hot(tf.cast(tf.squeeze(y, axis=-1), tf.int32), 10)
    return x, y

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = (train_ds
            .shuffle(10000)
            .batch(BATCH_SIZE)
            .map(lambda x, y: preprocess(x, y, training=True),
                 num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
           .batch(BATCH_SIZE)
           .map(lambda x, y: preprocess(x, y, training=False),
                num_parallel_calls=AUTOTUNE)
           .prefetch(AUTOTUNE))

# 3. EfficientNetB3 Fine-tuning 모델 구성
def build_efficientnet_model(trainable_base=False):
    # ImageNet 사전학습 가중치 로드, 최상단 분류기 제거
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = trainable_base  # 1단계: False, 2단계: True

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=trainable_base)

    # Classifier Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.Dropout(0.5)(x)  # 0.4 -> 0.5 (과적합 완화)
    outputs = layers.Dense(10, activation='softmax')(x)

    return models.Model(inputs, outputs)

# 4. Warmup + Cosine Decay 스케줄러
total_epochs_phase1 = 10   # 1단계: 헤드만 학습
total_epochs_phase2 = 30   # 2단계: 전체 Fine-tuning
warmup_epochs = 3
min_lr = 1e-6

def make_warmup_cosine_schedule(target_lr, total_epochs):
    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            return target_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return min_lr + (target_lr - min_lr) * cosine
    return schedule

# =============================================
# [1단계] Classifier Head만 학습 (base 동결)
# =============================================
print("\n" + "="*40)
print("[1단계] Head만 학습 (Base 동결)")
print("="*40)

model = build_efficientnet_model(trainable_base=False)
model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_efficientnet_phase1"

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_phase1 = [
    tf.keras.callbacks.LearningRateScheduler(
        make_warmup_cosine_schedule(target_lr=1e-3, total_epochs=total_epochs_phase1),
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5, update_freq='epoch'),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model_phase1.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history1 = model.fit(
    train_ds,
    epochs=total_epochs_phase1,
    validation_data=test_ds,
    callbacks=callbacks_phase1
)

# =============================================
# [2단계] 전체 Fine-tuning (base 해동)
# =============================================
print("\n" + "="*40)
print("[2단계] 전체 Fine-tuning (Base 해동)")
print("="*40)

# base_model 해동 (전체 레이어 학습 가능)
model.layers[1].trainable = True

# Fine-tuning은 lr을 훨씬 낮게 설정 (가중치 파괴 방지)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_efficientnet_phase2"

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_phase2 = [
    tf.keras.callbacks.LearningRateScheduler(
        make_warmup_cosine_schedule(target_lr=1e-4, total_epochs=total_epochs_phase2),
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5, update_freq='epoch'),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model_final.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history2 = model.fit(
    train_ds,
    epochs=total_epochs_phase2,
    validation_data=test_ds,
    callbacks=callbacks_phase2
)

# 5. 결과 분석
print("\n" + "="*30)
print("학습 완료! 최종 결과 분석")
print("="*30)

# 1단계 마지막 결과
p1_train_acc = history1.history['accuracy'][-1]
p1_val_acc   = history1.history['val_accuracy'][-1]
print(f"[1단계] Train: {p1_train_acc:.4f} / Val: {p1_val_acc:.4f}")

# 2단계 최종 결과
train_loss = history2.history['loss'][-1]
train_acc  = history2.history['accuracy'][-1]
val_loss, val_acc = model.evaluate(test_ds, verbose=0)

print(f"[2단계] Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
print(f"[2단계] Val   - Loss: {val_loss:.4f},  Accuracy: {val_acc:.4f}")
print("-" * 30)

gap = train_acc - val_acc
if gap > 0.1:
    print(f"진단: 격차 {gap:.4f} → 과적합. Dropout 강도를 높이거나 L2 정규화를 늘려보세요.")
elif gap < 0:
    print("진단: 언더피팅. 모델 용량을 늘리거나 에포크를 더 주세요.")
else:
    print(f"진단: 안정적인 학습 (격차: {gap:.4f}). 목표 95% 달성 여부를 확인하세요.")