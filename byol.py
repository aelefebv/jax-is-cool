import numpy as np

# Placeholder for your actual data loading logic
# data = np.load('your_data.npy')  # Shape: (T, C, H, W)

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

class ChannelWiseIntensityTransform(ImageOnlyTransform):
    def __init__(self, transforms, always_apply=False, p=1.0):
        super(ChannelWiseIntensityTransform, self).__init__(always_apply, p)
        self.transforms = transforms

    def apply(self, img, **params):
        channels = []
        for i in range(img.shape[0]):
            channel = img[i]
            augmented = self.transforms(image=channel)['image']
            channels.append(augmented)
        return np.stack(channels, axis=0)

# Geometric transformations (applied equally across channels)
geometric_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomResizedCrop(height=H, width=W, scale=(0.8, 1.0), p=0.5)
])

# Intensity transformations (applied differently per channel)
intensity_transforms = ChannelWiseIntensityTransform(
    transforms=A.Compose([
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
    ])
)

def augment(image):
    # Apply geometric transformations
    augmented = geometric_transforms(image=image.transpose(1, 2, 0))['image']
    augmented = augmented.transpose(2, 0, 1)
    # Apply intensity transformations per channel
    augmented = intensity_transforms(image=augmented)
    return augmented

def get_augmented_pair(data, t):
    """
    Get an augmented pair for time t and t+1 (if exists).

    :param data: NumPy array of shape (T, C, H, W)
    :param t: Time index
    :return: Tuple of augmented images
    """
    image_t = data[t]
    # Use adjacent timepoint if possible
    if t + 1 < data.shape[0]:
        image_t1 = data[t + 1]
    else:
        image_t1 = data[t - 1]  # Use previous timepoint if at the end

    # Augment both images separately
    augmented_view1 = augment(image_t)
    augmented_view2 = augment(image_t1)

    return augmented_view1, augmented_view2

import flax.linen as nn

class ResNetBlock(nn.Module):
    filters: int
    strides: int = 1

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(self.filters, (3, 3), self.strides)(x)
        y = nn.BatchNorm()(y)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3))(y)
        y = nn.BatchNorm()(y)

        if residual.shape != y.shape:
            residual = nn.Conv(self.filters, (1, 1), self.strides)(residual)
            residual = nn.BatchNorm()(residual)

        return nn.relu(residual + y)

class ResNet50(nn.Module):
    num_classes: int = 0  # Not used for BYOL

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (7, 7), strides=(2, 2))(x)
        x = nn.BatchNorm()(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        # Define blocks
        for filters, blocks, strides in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
            for i in range(blocks):
                x = ResNetBlock(filters, strides if i == 0 else 1)(x)

        x = nn.avg_pool(x, x.shape[1:3])
        x = x.reshape((x.shape[0], -1))
        return x  # Feature vector

import jax
import jax.numpy as jnp
from flax.training import moving_averages

class BYOL(nn.Module):
    encoder: nn.Module
    projector_dim: int = 256
    predictor_dim: int = 256

    def setup(self):
        self.online_encoder = self.encoder
        self.online_projector = nn.Dense(self.projector_dim)
        self.online_predictor = nn.Dense(self.predictor_dim)
        # Initialize target networks with the same parameters
        self.target_params = self.online_encoder.init(jax.random.PRNGKey(0), jnp.ones([1, C, H, W]))
        self.target_projector_params = self.online_projector.init(jax.random.PRNGKey(0), jnp.ones([1, F]))
        # EMA decay rate
        self.momentum = 0.99

    def __call__(self, x1, x2):
        # Online network
        z1 = self.online_encoder(x1)
        p1 = self.online_predictor(self.online_projector(z1))
        z2 = self.online_encoder(x2)
        p2 = self.online_predictor(self.online_projector(z2))
        # Target network (parameters are not updated by gradients)
        z1_target = self.online_encoder.apply({'params': self.target_params}, x1)
        z1_target = self.online_projector.apply({'params': self.target_projector_params}, z1_target)
        z2_target = self.online_encoder.apply({'params': self.target_params}, x2)
        z2_target = self.online_projector.apply({'params': self.target_projector_params}, z2_target)
        # Normalize outputs
        p1 = l2_normalize(p1)
        p2 = l2_normalize(p2)
        z1_target = l2_normalize(z1_target)
        z2_target = l2_normalize(z2_target)
        # Loss
        loss = byol_loss(p1, z2_target) + byol_loss(p2, z1_target)
        return loss

    def update_target(self):
        # Update target network parameters
        self.target_params = moving_averages.exponential_move_average(
            self.target_params, self.online_encoder.params, self.momentum
        )
        self.target_projector_params = moving_averages.exponential_move_average(
            self.target_projector_params, self.online_projector.params, self.momentum
        )

def l2_normalize(x, axis=None, epsilon=1e-12):
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
    return x * x_inv_norm

def byol_loss(p, z):
    return 2 - 2 * jnp.sum(p * z) / p.shape[0]

import optax

optimizer = optax.adam(learning_rate=1e-3)

@jax.jit
def train_step(state, x1, x2):
    def loss_fn(params):
        loss = state.apply_fn({'params': params}, x1, x2)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

from flax.training import train_state

# Initialize model
model = BYOL(encoder=ResNet50())
params = model.init(jax.random.PRNGKey(0), jnp.ones([1, C, H, W]), jnp.ones([1, C, H, W]))['params']

# Create train state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    for t in range(0, data.shape[0], batch_size):
        batch_indices = np.arange(t, min(t + batch_size, data.shape[0]))
        x1_batch = []
        x2_batch = []
        for idx in batch_indices:
            x1, x2 = get_augmented_pair(data, idx)
            x1_batch.append(x1)
            x2_batch.append(x2)
        x1_batch = np.stack(x1_batch)
        x2_batch = np.stack(x2_batch)
        # Convert to JAX arrays
        x1_batch = jnp.array(x1_batch)
        x2_batch = jnp.array(x2_batch)
        # Perform a training step
        state = train_step(state, x1_batch, x2_batch)
        # Update target network
        model.update_target()
    print(f"Epoch {epoch + 1} completed.")

import flax.serialization

# Save parameters
with open('model_checkpoint.pkl', 'wb') as f:
    f.write(flax.serialization.to_bytes(state.params))

# Load parameters
with open('model_checkpoint.pkl', 'rb') as f:
    params = flax.serialization.from_bytes(state.params, f.read())
    state = state.replace(params=params)
