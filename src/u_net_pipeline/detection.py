import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import (Input, Conv2D, MaxPool2D,
                                     Conv2DTranspose, concatenate)
from keras.models import Model
from keras.optimizers import Adam

# Paths
data_raw = 'data/raw'
data_masks = 'data/labels'     # assume binary masks here
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 512
BATCH_SIZE = 8
EPOCHS = 20

# U-Net builder
def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = Input(input_shape)
    # Encoder
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPool2D()(c1)

    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPool2D()(c2)

    # Bottleneck
    b = Conv2D(128, 3, activation='relu', padding='same')(p2)
    b = Conv2D(128, 3, activation='relu', padding='same')(b)

    # Decoder
    u2 = Conv2DTranspose(64, 2, strides=2, padding='same')(b)
    u2 = concatenate([u2, c2])
    c3 = Conv2D(64, 3, activation='relu', padding='same')(u2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

    u3 = Conv2DTranspose(32, 2, strides=2, padding='same')(c3)
    u3 = concatenate([u3, c1])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(u3)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)

    outputs = Conv2D(1, 1, activation='sigmoid')(c4)
    return Model(inputs, outputs)

# Load video frames and corresponding mask frames
def load_dataset():
    video_paths = glob.glob(os.path.join(data_raw, '*.mp4')) + \
                  glob.glob(os.path.join(data_raw, '*.mov'))
    X, Y = [], []
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        fname = os.path.splitext(os.path.basename(vp))[0]
        # assume mask video named similarly in data/labels
        mask_path = os.path.join(data_masks, f"{fname}.mp4")
        mcap = cv2.VideoCapture(mask_path)
        while True:
            ret, frame = cap.read()
            mret, mframe = mcap.read()
            if not ret or not mret:
                break
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            mframe = cv2.cvtColor(cv2.resize(mframe, (IMG_WIDTH, IMG_HEIGHT)), cv2.COLOR_BGR2GRAY)
            mframe = (mframe > 127).astype(np.float32)
            X.append(frame / 255.0)
            Y.append(np.expand_dims(mframe, -1))
        cap.release(); mcap.release()
    return np.array(X), np.array(Y)

# Main
def main():
    print("Loading dataset...")
    X, Y = load_dataset()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    print("Building model...")
    model = build_unet()
    model.compile(optimizer=Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("Training...")
    model.fit(X_train, Y_train,
              validation_data=(X_test, Y_test),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS)

    print("Testing and saving results...")
    preds = model.predict(X_test)
    for i, (img, pred) in enumerate(zip(X_test, preds)):
        mask = (pred[...,0] > 0.5).astype(np.uint8) * 255
        overlay = cv2.addWeighted((img*255).astype(np.uint8),
                                  0.8,
                                  cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                                  0.2, 0)
        cv2.imwrite(os.path.join(results_dir, f"result_{i}.png"), overlay)

    model.save(os.path.join(results_dir, 'unet_lane_model.h5'))
    print("Done. Outputs saved in results/")

if __name__ == '__main__':
    main()
