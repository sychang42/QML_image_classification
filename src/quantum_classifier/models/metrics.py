r"""
Losses used for quantum classifier training.
"""
import jax
import jax.numpy as jnp

from jax import Array

from typing import Tuple, Dict, List


def MSE(x: Array, y: Array) -> Array:
    r"""Measures the Mean Squared Error (MSE) loss between each element in the target
    :math:`x` and the input :math:`y` given by the equation :

    .. math::
        \ell_{\text{MSE}}(x, y) = \frac{1}{N}\sum_{n=1}^N\sqrt{\left( x_n - y_n \right)^2},

    where :math:`N` is the number of elements in :math:`x` and :math:`y`.

    Args:
        x (Array): Targets of shape ``(N, 1)``
        y (Array): Inputs of shape ``(N, 1)``.

    Returns:
        Array: MSE loss value.
    """
    return jnp.mean((x - y) ** 2)


def BCE_loss(x: Array, y: Array) -> Array:
    r"""Measures the Binary Cross Entropy (BCE) loss between each element in the one-hot
    encoded target :math:`x` and the input :math:`y` given by the equations :


    .. math::
        \ell(x, y) = - \sum_{n=1}^N_\mathbf{x}_n \cdot \log (\mathbf{y}_n),

    where :math:`N` is the batch size.

    Args:
        x (Array): Targets of shape ``(N, L)``
        y (Array): Targets of shape ``(N, L)``

    Returns:
        Array: BCE loss value.
    """
    num_classes = y.shape[1]

    one_hot_labels = jax.nn.one_hot(x, num_classes)
    return -jnp.mean(jnp.sum(one_hot_labels * jnp.log(y), axis=-1))


def accuracy(targets: Array, class_outputs: Array) -> Array:
    """_summary_

    Args:
        targets (Array): _description_
        class_outputs (Array): _description_

    Returns:
        Array: Accuracy caculated between ``targets`` and ``class_outputs``.
    """
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
    r"""Compute the

    Args:
        loss_type (List[str]): List of strings representing the loss types that are
            computed.
        targets (Array): The ground truth labels.
        preds (Array): The predicted labels.

    Returns:
        Tuple[Array, Dict[str, Array]]: A tuple containing the sum of losses that on
        which the gradient of the parameters is computed, and a dictionary of individual
        losses.
    """
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

    return final_loss, losses  # type: ignore
