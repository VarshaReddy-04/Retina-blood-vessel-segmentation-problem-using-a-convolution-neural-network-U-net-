from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def unet(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    # Bottleneck
    b = Conv2D(256, 3, activation='relu', padding='same')(p2)
    b = Conv2D(256, 3, activation='relu', padding='same')(b)

    # Decoder
    u1 = UpSampling2D()(b)
    u1 = concatenate([u1, c2])
    c3 = Conv2D(128, 3, activation='relu', padding='same')(u1)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)

    u2 = UpSampling2D()(c3)
    u2 = concatenate([u2, c1])
    c4 = Conv2D(64, 3, activation='relu', padding='same')(u2)
    c4 = Conv2D(64, 3, activation='relu', padding='same')(c4)

    outputs = Conv2D(1, 1, activation='sigmoid')(c4)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
