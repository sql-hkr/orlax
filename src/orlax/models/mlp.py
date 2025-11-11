"""Multi-layer perceptron (MLP) implementation in Flax.

Simple MLP building block for policy and value networks.
"""

from typing import Sequence, Callable
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture.
    
    Attributes:
        hidden_dims: Sequence of hidden layer dimensions
        output_dim: Output dimension (None to omit final layer)
        activation: Activation function name or callable
        dropout_rate: Dropout rate (0.0 to disable)
        use_layer_norm: Whether to use layer normalization
        use_bias: Whether to use bias in linear layers
        
    Example:
        model = MLP(
            hidden_dims=(256, 256),
            output_dim=10,
            activation='relu'
        )
        output = model(x)
    """
    
    hidden_dims: Sequence[int]
    output_dim: int | None = None
    activation: str | Callable = "relu"
    dropout_rate: float = 0.0
    use_layer_norm: bool = False
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            x: Input array [batch_size, input_dim]
            training: Whether in training mode (for dropout)
            
        Returns:
            Output array [batch_size, output_dim]
        """
        # Get activation function
        if isinstance(self.activation, str):
            activation_fn = _get_activation(self.activation)
        else:
            activation_fn = self.activation
        
        # Hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                hidden_dim,
                use_bias=self.use_bias,
                name=f"hidden_{i}"
            )(x)
            
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            
            x = activation_fn(x)
            
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output layer (optional)
        if self.output_dim is not None:
            x = nn.Dense(self.output_dim, use_bias=self.use_bias, name="output")(x)
        
        return x


def _get_activation(name: str) -> Callable:
    """Get activation function by name.
    
    Args:
        name: Activation function name
        
    Returns:
        Activation function
    """
    activations = {
        "relu": nn.relu,
        "tanh": nn.tanh,
        "gelu": nn.gelu,
        "elu": nn.elu,
        "silu": nn.silu,
        "swish": nn.swish,
        "sigmoid": nn.sigmoid,
        "softplus": nn.softplus,
    }
    
    if name not in activations:
        raise ValueError(
            f"Unknown activation: {name}. "
            f"Available: {list(activations.keys())}"
        )
    
    return activations[name]
