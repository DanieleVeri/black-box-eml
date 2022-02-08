import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.quantization.keras import quantize_annotate_model
from .quantizer import download_pretrained_model, QScheme

def build_tpc():
    cifar10 = tf.keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    dataset_mean = train_images.mean(axis=(0,1,2))
    dataset_std = train_images.std(axis=(0,1,2))
    train_images = (train_images - dataset_mean) / dataset_std
    test_images = (test_images - dataset_mean) / dataset_std
    download_pretrained_model()

    def tpc(x):
        print("Query for x=", x)
        model = tf.keras.models.load_model("pretrained_resnet18")
        model = quantize_annotate_model(model)
        q_aware_model = tfmot.quantization.keras.quantize_apply(model, QScheme(x))
        q_aware_model.compile(optimizer="adam",
                        loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        # q_aware_model.summary()

        q_aware_model.fit(train_images, train_labels,
                        batch_size=512, epochs=5, validation_split=0.1,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                                monitor="val_loss", 
                                patience=5, 
                                restore_best_weights=True)])
        
        q_aware_model_loss, q_aware_model_accuracy = q_aware_model.evaluate(
            test_images, test_labels, verbose=0)
        
        return -q_aware_model_accuracy
    return tpc

def constraint_max_bits(xvars):
    return [[sum(xvars) <= len(xvars)*2.5, 'cst_size']]