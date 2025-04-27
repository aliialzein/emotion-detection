import sys
sys.path.append('/content/drive/MyDrive/emotion-detection/')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from utils.preprocess import create_data_generators  # Assuming you call the function from preprocess

# ========== DATA PATH ==========
train_dir = '/content/drive/MyDrive/emotion-detection/data/train'
test_dir = '/content/drive/MyDrive/emotion-detection/data/test'
IMG_SIZE = (96, 96)  # Bigger input now
BATCH_SIZE = 32
NUM_CLASSES = 7  # Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

# ========== DATA GENERATORS ==========
train_generator, val_generator, test_generator = create_data_generators(train_dir, test_dir, IMG_SIZE, BATCH_SIZE)

# ========== CLASS WEIGHTS ==========
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# ========== BUILD MODEL ==========
def build_model(input_shape, num_classes):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False  # Freeze base at start

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model  # ✅ return both model and base_model

# Build model
model, base_model = build_model((96, 96, 3), NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ========== CALLBACKS ==========
if not os.path.exists('models'):
    os.makedirs('models')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/emotion_model.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# ========== INITIAL TRAINING ========== 
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weights
)

print("✅ Initial training finished! Now starting fine-tuning...")

# ========== FINE-TUNING ==========
# Unfreeze the base model (MobileNetV2 only)
base_model.trainable = True

# Re-compile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set fine-tuning parameters
fine_tune_epochs = 20
total_epochs = 30 + fine_tune_epochs

# Fine-tune the model
history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=30,  # ✅ start from where you left off (30)
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weights
)

print("✅ Fine-tuning finished! Final model saved to 'models/emotion_model.keras'.")
