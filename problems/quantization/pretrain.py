import wandb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, Input, ReLU, Activation
from emlopt.utils import set_seed



def resnet_block(channels, down_sample=False):
    strides = [2, 1] if down_sample else [1, 1]
    KERNEL_SIZE = (3, 3)
    INIT_SCHEME = "he_normal"

    relu1 = ReLU()
    relu2 = ReLU()

    conv_1 = Conv2D(channels, strides=strides[0],
                       kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
    bn_1 = BatchNormalization()
    conv_2 = Conv2D(channels, strides=strides[1],
                       kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
    bn_2 = BatchNormalization()
    merge = Add()

    if down_sample:
        # perform down sampling using stride of 2, according to [1].
        res_conv = Conv2D(
            channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
        res_bn = BatchNormalization()

    def call(inputs):
        res = inputs
        x = conv_1(inputs)
        x = bn_1(x)
        x = Activation('relu')(x)
        x = conv_2(x)
        x = bn_2(x)

        if down_sample:
            res = res_conv(res)
            res = res_bn(res)
        # if not perform down sample, then add a shortcut directly
        x = merge([x, res])
        out = Activation('relu')(x)
        return out

    return call

def build_resnet():
    ks = 64
    inputs = Input(shape=(32, 32, 3))
    
    relu1 = ReLU()
    conv_1 = Conv2D(ks, (3, 3), strides=1,
                       padding="same", kernel_initializer="he_normal")
    init_bn = BatchNormalization()
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
    res_1_1 = resnet_block(ks)
    res_1_2 = resnet_block(ks)
    res_2_1 = resnet_block(ks*2, down_sample=True)
    res_2_2 = resnet_block(ks*2)
    res_3_1 = resnet_block(ks*4, down_sample=True)
    res_3_2 = resnet_block(ks*4)
    res_4_1 = resnet_block(ks*8, down_sample=True)
    res_4_2 = resnet_block(ks*8)

    avg_pool = GlobalAveragePooling2D()
    flat = Flatten()
    fc = Dense(10, activation="softmax")

    out = conv_1(inputs)
    out = init_bn(out)
    out = Activation('relu')(out)
    out = pool_2(out)
    for res_block in [res_1_1, res_1_2, res_2_1, res_2_2, res_3_1, res_3_2, res_4_1, res_4_2]:
        out = res_block(out)
    out = avg_pool(out)
    out = flat(out)
    out = fc(out)

    return Model(inputs, out)

def pretrain(project_name: str):
    set_seed()

    # Load CIFAR10 dataset
    cifar10 = tf.keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    dataset_mean = train_images.mean(axis=(0,1,2))
    dataset_std = train_images.std(axis=(0,1,2))

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = (train_images - dataset_mean) / dataset_std
    test_images = (test_images - dataset_mean) / dataset_std

    # Build float model
    model = build_resnet()
    model.compile(optimizer="adam",
                    loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    model.summary()

    # Train float model
    model.fit(
        train_images,
        train_labels,
        batch_size=128,
        epochs=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5, restore_best_weights=True)],
        validation_split=0.1)

    # Store model artifact
    model.save("pretrained_resnet18")
    artifact = wandb.Artifact('resnet18', type='model')
    artifact.add_dir("pretrained_resnet18")
    wandb.init(project=project_name, name="artifact_resnet18")
    wandb.log_artifact(artifact)
    wandb.finish()