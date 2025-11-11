"""Optimizer creation and configuration using Optax."""

import optax


def create_optimizer(
    learning_rate: float,
    optimizer_type: str = "adam",
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    **kwargs
) -> optax.GradientTransformation:
    """Create an Optax optimizer.
    
    Args:
        learning_rate: Learning rate
        optimizer_type: Optimizer type ('adam', 'adamw', 'sgd', 'rmsprop')
        weight_decay: Weight decay coefficient
        max_grad_norm: Maximum gradient norm for clipping
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optax gradient transformation (optimizer)
        
    Example:
        optimizer = create_optimizer(learning_rate=3e-4, optimizer_type="adam")
    """
    # Select base optimizer
    if optimizer_type == "adam":
        opt = optax.adam(learning_rate, **kwargs)
    elif optimizer_type == "adamw":
        opt = optax.adamw(learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == "sgd":
        opt = optax.sgd(learning_rate, **kwargs)
    elif optimizer_type == "rmsprop":
        opt = optax.rmsprop(learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Add gradient clipping
    if max_grad_norm > 0:
        opt = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            opt
        )
    
    return opt


def create_lr_schedule(
    base_lr: float,
    schedule_type: str = "constant",
    warmup_steps: int = 0,
    total_steps: int = 1_000_000,
    **kwargs
) -> optax.Schedule:
    """Create a learning rate schedule.
    
    Args:
        base_lr: Base learning rate
        schedule_type: Schedule type ('constant', 'cosine', 'linear', 'exponential')
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        **kwargs: Additional schedule-specific arguments
        
    Returns:
        Optax learning rate schedule
        
    Example:
        schedule = create_lr_schedule(3e-4, 'cosine', warmup_steps=10000, total_steps=1000000)
    """
    if schedule_type == "constant":
        schedule = optax.constant_schedule(base_lr)
    elif schedule_type == "cosine":
        if warmup_steps > 0:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=base_lr,
                warmup_steps=warmup_steps,
                decay_steps=total_steps - warmup_steps,
                end_value=0.0,
            )
        else:
            schedule = optax.cosine_decay_schedule(
                init_value=base_lr,
                decay_steps=total_steps,
                alpha=0.0,
            )
    elif schedule_type == "linear":
        if warmup_steps > 0:
            warmup = optax.linear_schedule(0.0, base_lr, warmup_steps)
            decay = optax.linear_schedule(base_lr, 0.0, total_steps - warmup_steps)
            schedule = optax.join_schedules([warmup, decay], [warmup_steps])
        else:
            schedule = optax.linear_schedule(base_lr, 0.0, total_steps)
    elif schedule_type == "exponential":
        decay_rate = kwargs.get("decay_rate", 0.99)
        transition_steps = kwargs.get("transition_steps", 10000)
        schedule = optax.exponential_decay(
            init_value=base_lr,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return schedule
