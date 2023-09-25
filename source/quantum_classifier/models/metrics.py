import jax
import jax.numpy as jnp

from jax import Array

from typing import Tuple, Dict, List


def MSE(targets: Array, preds: Array) -> Array:
    return jnp.mean((targets - preds) ** 2)


def BCE_loss(targets: Array, preds: Array) -> Array:
    num_classes = preds.shape[1]

    one_hot_labels = jax.nn.one_hot(targets, num_classes)
    return -jnp.mean(jnp.sum(one_hot_labels * jnp.log(preds), axis=-1))


def accuracy(targets: Array, class_outputs: Array) -> Array:
    if len(class_outputs[0]) > 1:
        preds = jnp.argmax(class_outputs, -1)
    else:
        preds = jnp.round(class_outputs)

    corrects = jnp.asarray((targets == preds), dtype=float)
    acc = jnp.sum(corrects) / len(corrects)

    return acc


def compute_metrics(
    loss_type: List[str], targets: Array, preds: Array
) -> Tuple[Array, Dict[str, Array]]:
    losses: Dict[str, Array] = {}
    final_loss = 0.0

    for loss_str in loss_type:
        switcher = {
            "MSE_loss": MSE,
            "BCE_loss": BCE_loss,
            "accuracy": accuracy,
        }
        loss_fn = switcher.get(loss_str, None)

        if loss_fn is None:
            break

        loss = loss_fn(targets, preds)
        losses[loss_str] = loss

        if "loss" in loss_str:
            final_loss += loss

    return final_loss, losses
