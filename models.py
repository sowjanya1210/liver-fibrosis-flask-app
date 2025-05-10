import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
import tensorflow.keras.backend as K



@tf.keras.utils.register_keras_serializable()
def sampling(args):
    """Reparameterization trick to sample latent vector z."""
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def create_vae_cnn_model(input_shape=(128, 128, 3), num_classes=5, latent_dim=64):
    """Creates a Variational Autoencoder (VAE) with CNN-based encoder, decoder, and classifier."""

    ### 🔹 Encoder ###
    inputs = Input(shape=input_shape, name="input_layer")
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    # Sample z using reparameterization trick
    # Instead of using the function directly, reference it by name (optional but ensures consistency)
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name="latent_space")([z_mean, z_log_var])

    ### 🔹 Decoder (Reconstructs Input) ###
    x = layers.Dense((input_shape[0] // 8) * (input_shape[1] // 8) * 128, activation='relu')(z)
    x = layers.Reshape((input_shape[0] // 8, input_shape[1] // 8, 128))(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoder_output = layers.Conv2DTranspose(input_shape[2], (3, 3), activation='sigmoid', padding='same', name="decoder_output")(x)

    ### 🔹 Classifier (Predicts Class) ###
    classifier_output = layers.Dense(num_classes, activation='softmax', name="classifier_output")(z)

    ### 🔹 VAE Model ###
    model = Model(inputs=inputs, outputs=[decoder_output, classifier_output], name="VAE_CNN_Classifier")

    ### 🔹 Custom Loss (Reconstruction + KL Divergence) ###
    def vae_loss(y_true, y_pred):
        recon_loss = MeanSquaredError()(y_true, y_pred)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return recon_loss + 0.0001 * kl_loss  # Small weight for KL loss

    ### 🔹 Compile Model ###
    model.compile(
        optimizer='adam',
        loss={
            "decoder_output": vae_loss,  # Custom VAE Loss
            "classifier_output": CategoricalCrossentropy()  # Classification Loss
        },
        loss_weights={"decoder_output": 1.0, "classifier_output": 1.0},  # Equal importance
        metrics={"classifier_output": "accuracy"}
    )

    return model

# Create the model
vae_model = create_vae_cnn_model()
vae_model.summary()