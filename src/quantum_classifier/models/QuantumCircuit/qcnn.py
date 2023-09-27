"""
Quantum Convolutional Neural Network. 
"""
import sys
import os

sys.path.append(os.path.dirname(__file__))

import json

import jax.numpy as jnp
from jax import Array

import numpy as np
import pennylane as qml
import unitary
from math import ceil
from typing import Callable, Tuple, Optional


# Valid quantum convolutional filters
_valid_gates = {
    "RZ": (qml.RZ, 2, 2),
    "U_TTN": (unitary.U_TTN, 2, 2),
    "U_5": (unitary.U_5, 10, 2),
    "U_6": (unitary.U_6, 10, 2),
    "U_9": (unitary.U_9, 2, 2),
    "U_13": (unitary.U_13, 6, 2),
    "U_14": (unitary.U_14, 6, 2),
    "U_15": (unitary.U_15, 4, 2),
    "U_SO4": (unitary.U_SO4, 6, 2),
    "U_SU4": (unitary.U_SU4, 15, 2),
    "U_ZZ": (unitary.U_ZZ, 15, 2),
    "U_qiskit": (unitary.U_qiskit, 15, 2),
    "U_RX": (unitary.U_RX, 2, 2),
    "Pooling_ansatz1": (unitary.Pooling_ansatz1, 2, 2),
}


def choose_gate(gate_str: str) -> Tuple[str, Callable, int, int]:
    r"""Helper function to used to retrieve a specified convolutional filter (gate).

    Args:
        gate_str (str): Name of the convolutional filter to be loaded.

    Returns:
        Tuple[str, Callable, int, int]: Tuple containing the name of the convolutional
        filter (given as args), the function representing the convolutional filter,
        the number of parameters in the filter and the number of wires on which the
        gate is applied.

    Example:

        >>> gate = choose_gate("U_TTN")
        >>> print(gate)
            ('U_TTN', <function unitary.U_TTN(angle, wires)>, 2, 2)

    """
    gate = _valid_gates.get(gate_str, None)

    if gate is None:
        raise NotImplementedError("Unknown gate.")

    return (gate_str, gate[0], gate[1], gate[2])


def U2_conv(params: Array, gate: Callable, wires: list[int]) -> None:
    r"""Convolutional filter applied on two qubits.

    Args :
        params (Array) : Convolutional filter parameters.
        gate (Callabel) : Callable representing convolutional filter.
        wires (list[int]) : Index of wires where the gate will be applied.
    """
    idx = 0
    for i in range(0, len(wires), 2):
        gate(params[idx], wires=[wires[i], wires[i + 1]])
        idx = idx + 1

    for i in range(1, len(wires) - 1, 2):
        gate(params[idx], wires=[wires[i], wires[i + 1]])
        idx = idx + 1

    gate(params[idx], wires=[wires[-1], wires[0]])


def QCNN(
    num_qubits: int,
    num_measured: int,
    trans_inv: Optional[bool],
    qnn_ver: Optional[str] = None,
) -> Tuple[Callable, int, np.ndarray]:
    r"""Construct Quantum Convolutional Neural Network architecture uing the specified
    QCNN version.

    Args :
        num_qubits (int) : Number of qubits in the QCNN.
        num_measured (int) : Number of measured qubits at the end of the circuit.
            For L classes, we measure ceil(log2(L)) qubits.
        trans_inv (bool, optional) : Boolean to indicate whether the QCNN is
            translational invariant or not. If True, all filters in a layer share
            identical parameters; otherwise, different parameters are used. (To be
            implemented)
        qnn_ver (str, optional) : Version of the quantum circuit architecture to be
                            used. If set to None, the default architecture with U_TTN
                            convolutional filters is used.

    Returns :
        Tuple[Callable, int, np.ndarray]: Return a functionrepresenting the QCNN circuit,
        the total number of parameters in the circuit, and the list of wires measurment
        at the end of the circuit.
    """
    qnn_config_path = os.path.join(os.path.dirname(__file__), "qnn_architecture.json")

    # Default QNN architecture
    qnn_architecture = {"conv_filters": ["U_TTN"], "pooling": "Pooling_ansatz1"}
    if qnn_ver is not None:
        qnn_architecture = json.load(open(qnn_config_path))[qnn_ver]

    conv_filters = []
    if "conv_filters" in qnn_architecture.keys():
        conv_filters = [choose_gate(gate) for gate in qnn_architecture["conv_filters"]]

    pooling = []
    if "pooling" in qnn_architecture.keys():
        pooling = choose_gate(qnn_architecture["pooling"])

    depth = ceil(np.log2(num_qubits // num_measured))
    meas_wires = [i for i in range(num_qubits // 2)]

    while len(meas_wires) > num_measured:
        meas_wires = [meas_wires[i] for i in range(0, len(meas_wires), 2)]

    meas_wires = np.array(meas_wires)

    num_params = depth * (sum([gate[2] for gate in conv_filters]) + pooling[2])

    def circuit(params: Array) -> None:
        idx = 0

        wires = np.array([i for i in range(num_qubits)])

        while len(wires) > num_measured:
            for _, gate, num_params, gate_num_wires in conv_filters:
                for i in range(0, len(wires), 2):
                    gate(params[idx : idx + num_params], wires=[wires[i], wires[i + 1]])
                for i in range(1, len(wires) - 1, 2):
                    gate(params[idx : idx + num_params], wires=[wires[i], wires[i + 1]])

                gate(params[idx : idx + num_params], wires=[wires[-1], wires[0]])

                idx = idx + num_params

            _, gate, num_params, gate_num_wires = pooling

            traced_out_wires = []

            if len(wires) > 2:
                for i in range(0, len(wires) // 2 - 1, 2):
                    gate(params[idx : idx + num_params], wires=[wires[i], wires[i + 1]])
                    traced_out_wires.append(i + 1)

                for i in range(len(wires) // 2, len(wires) - 1, 2):
                    gate(params[idx : idx + num_params], wires=[wires[i], wires[i + 1]])
                    traced_out_wires.append(i + 1)
            else:
                for i in range(0, len(wires), 2):
                    gate(params[idx : idx + num_params], wires=[wires[i], wires[i + 1]])
                    traced_out_wires.append(i + 1)

            idx = idx + num_params

            wires = np.delete(wires, traced_out_wires)

    return circuit, num_params, meas_wires
