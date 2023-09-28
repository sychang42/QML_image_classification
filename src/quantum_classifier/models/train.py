"""
Training functions for the quantum classifier. 
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))

from time import time

import csv
from typing import Callable, Dict, List, Tuple, Any, Optional

import jax
import jax.numpy as jnp
from jax import Array


import optax
import pennylane as qml
from metrics import compute_metrics
from QuantumCircuit.embedding import get_data_embedding
from QuantumCircuit.qcnn import QCNN
from tqdm import tqdm
from utils import save_outputs, print_losses


def train_batch(
    x_batch: Array,
    y_batch: Array,
    loss_type: List,
    circuit: Callable,
    class_params: optax.Params,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> Tuple[optax.OptState, optax.Params, Dict[str, Array], Array]:
    r"""Train the quantum classifier model on a batch.

    Args:
        x_batch (Array): Batch of training data.
        y_batch (Array): Batch of training data labels.
        loss_type (List): List of strings representing the loss types used in the
            training.
        circuit (Callable): Callable representing the quantum classifier circuit.
        class_params (optax.Params): Quantum circuit parameters.
        optimizer (optax.GradientTransformation): Optimizer used for training.
        opt_state (optax.OptState): Optimizer state which is going to be updated.

    Returns:
        Tuple[optax.OptState, optax.Params, Dict[str, Array], Array]: A tuple containing
        updated optimizer state, updated training parameters, calculated loss, and the
        classifier outputs for the input data ``x_batch``.
    """

    def loss_fn(params) -> Tuple[Array, Tuple[Dict[str, Array], Array]]:
        vcircuit = jax.vmap(lambda x: circuit(x, params))

        class_outputs = vcircuit(x_batch)
        loss, losses = compute_metrics(loss_type, y_batch, class_outputs)
        return loss, (losses, class_outputs)

    grads, (losses, class_outputs) = jax.grad(loss_fn, has_aux=True)(class_params)

    updates, opt_state = optimizer.update(grads, opt_state)
    class_params = optax.apply_updates(class_params, updates)
    return opt_state, class_params, losses, class_outputs


def validate(
    x_batch: Array,
    y_batch: Array,
    loss_type: List,
    circuit: Callable,
    class_params: Array,
) -> Tuple[Dict[str, Array], Any]:
    r"""Evaluate the validation loss for the model.


    Args:
        x_batch (Array): A batch of validation data.
        y_batch (Array): A batch of validation labels.
        loss_type (List): List of strings representing the loss types used in the
            training.
        circuit (Callable): A runction representig the quantum classifier.
        class_params (Array): Trained parameters of quantum classifier.

    Returns:
        Tuple[Dict[str, Array], Array]:  A tuple containing a dictionary with validation
        loss values and the predicted class labels for the test set.
    """

    vcircuit = jax.vmap(lambda x: circuit(x, class_params))

    class_outputs = vcircuit(x_batch)
    loss, losses = compute_metrics(loss_type, y_batch, class_outputs)

    if class_outputs.shape[1] == 1:
        preds = jnp.round(class_outputs).tolist()
    else:
        preds = jnp.argmax(class_outputs, axis=1).tolist()
    return losses, preds


def train(
    train_ds: Dict[str, Array],
    test_ds: Dict[str, Array],
    train_args: Dict[str, Any],
    model_args: Dict[str, Any],
    optim_args: Dict[str, float],
    snapshot_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    r"""Train the quantum classifier using the provided datasets and hyperparameters.

    Args:
        train_ds (Dict[str, Array]): The training dataset.
        test_ds (Dict[str, Array]): The test dataset.
        train_args (Dict[str, Any]): Hyperparameters needed for classifier training.
            The dictionary should contain the following points:

            * *num_epochs*: Integer indicating the number of training epochs.
            * *batch_size*: Integer indicating the training batch size.
            * *loss_type*: List of strings representing the loss types used in the
                training. Currently, only :func:`metrics.BCE_loss` and
                :func:`metrics.accuracy` are supported.
            * *seed*: Seed used to generate random values using JAX's random number
                generator (:class:`jax.random`).

        model_args (Dict[str, Any]): Arguments required to constructed the QCNN.

            * *num_wires*: Number of qubits in the QCNN.
            * *num_measured*: Number of measured qubits at the end of the circuit.For
                L classes, we measure :math:`\lceil (\log_2(L))\rceil` qubits.
            * *trans_inv*: Boolean to indicate whether the QCNN is
                translational invariant or not. If True, all filters in a layer share
                identical parameters; otherwise, different parameters are used. (To be
                implemented)

        optim_args (Dict[str, float]): :class:`optax.adam` optimizer hyperparameters.

            * *learning_rate*: Learning rate for the optimizer.
            * *b1*: :math:`\beta_1` value of the Adam optimizer.
            * *b2*: :math:`\beta_2` value of the Adam optimizer.

        snapshot_dir (str, optional): Directory to store the training result if not
            None. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Tuple of dictionaries containing the
        training and test set loss progression based on the specified ``loss_type``.

    Example:

        >>> train_args = {"num_epochs": 5, "batch_size" : 1024, "loss_type": ['BCE_loss',
        ...               'accuracy]}
        >>> model_args = {"num_wires": 8, "num_measured" : 1, "trans_inv": True}
        >>> opt_args = {"learning_rate": 0.01, "b1" : 0.9, "b2": 0.999}
        >>> train_loss, test_loss = train(train_ds, test_ds, train_args, model_args,
        ...                         opt_args, "Result")
        >>> print(train_loss)
            {'BCE_loss': [0.6449261327584586, 0.600408172607422, 0.5616568744182586,
            0.5470444520314535, 0.537257703145345], 'accuracy': [0.6750000019868214,
            0.7497685273488361, 0.8037037114302319, 0.8217592616875966, 0.8263888935248058]
            }

    """
    num_epochs = train_args["num_epochs"]
    batch_size = train_args["batch_size"]
    loss_type = train_args["loss_type"]
    seed = train_args["seed"]

    optimizer = optax.adam(
        optim_args["learning_rate"], b1=optim_args["b1"], b2=optim_args["b2"]
    )

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    embed_data = get_data_embedding(model_args["Embedding"])
    qcnn_circuit, num_params, meas_wires = QCNN(
        model_args["num_wires"],
        model_args["num_measured"],
        model_args["trans_inv"],
        model_args["ver"],
    )

    def classifier_circuit(
        X: Array, params: optax.Params
    ) -> list[qml.measurements.ExpectationMP] | qml.measurements.ProbabilityMP:
        embed_data(X)
        qcnn_circuit(params)

        if "MSE_loss" in loss_type:
            return [qml.expval(qml.PauliZ(i)) for i in meas_wires]
        elif "BCE_loss" in loss_type:
            return qml.probs(meas_wires)
        else:
            raise NotImplementedError("Error type not suppoerted.")

    params = jax.random.normal(rng, (num_params,))
    rng, init_rng = jax.random.split(rng)
    opt_state = optimizer.init(params)

    dev = qml.device("default.qubit", wires=model_args["num_wires"])

    qnode_circuit = qml.QNode(classifier_circuit, dev, interface="jax-jit")
    qcircuit = jax.jit(qnode_circuit)
    # Training of model
    train_losses = {k: [] for k in loss_type}
    test_losses = {k: [] for k in loss_type}

    train_ds_size = len(train_ds["image"])
    print("num_training: ", train_ds_size)
    print(test_ds["image"].shape)

    steps_per_epoch = train_ds_size // batch_size

    if snapshot_dir is not None:
        with open(os.path.join(snapshot_dir, "output.csv"), mode="w") as csv_file:
            fieldnames = ["epoch"]
            fieldnames.extend(["train_" + k for k in loss_type])
            fieldnames.extend(["test_" + k for k in loss_type])
            fieldnames.append("time_taken")
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(num_epochs):
        start_time = time()

        rng, init_rng = jax.random.split(rng)
        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))

        for k in train_losses.keys():
            train_losses[k].append(0.0)

        with tqdm(perms, unit="batch") as tepoch:
            for perm in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                batch = {k: v[perm, ...] for k, v in train_ds.items()}

                opt_state, params, losses, class_outputs = train_batch(
                    batch["image"],
                    batch["label"],
                    loss_type,
                    qcircuit,
                    params,
                    optimizer,
                    opt_state,
                )

                for k, v in losses.items():
                    train_losses[k][-1] += v.tolist() / steps_per_epoch

        test_loss, test_predictions = validate(
            test_ds["image"], test_ds["label"], loss_type, qcircuit, params  # type: ignore
        )
        for k, v in test_loss.items():
            test_losses[k].append(v.tolist())

        # Save results.
        if snapshot_dir is not None:
            # Store output
            with open(os.path.join(snapshot_dir, "output.csv"), mode="a") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                to_write = {"epoch": epoch}
                to_write.update({"train_" + k: v[-1] for k, v in train_losses.items()})
                to_write.update({"test_" + k: v for k, v in test_loss.items()})  # type: ignore
                to_write["time_taken"] = (time() - start_time) / 60.0  # type: ignore

                writer.writerow(to_write)

            with open(
                os.path.join(snapshot_dir, "train_parameters.txt"), mode="a"
            ) as f:
                for x in params:
                    f.write(str(x) + " ")
                f.write("\n")
            if epoch == 1 or epoch % 10 == 0:
                save_outputs(epoch, snapshot_dir, test_predictions, test_ds["label"])

        print_losses(
            epoch, num_epochs, {k: v[-1] for k, v in train_losses.items()}, test_loss  # type: ignore
        )

    return train_losses, test_losses
