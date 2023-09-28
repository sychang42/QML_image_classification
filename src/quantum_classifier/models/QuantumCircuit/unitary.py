"""
This module contains the set of unitary ansatze that are used as convolutional filters
in :func:`qcnn.QCNN`. Some of them are defined following the definition given in Ref.
[1].
    
    **References** 
    [1]: T. Hur, L. Kim, and D. K. Park, `Quantum convolutional neural network for 
    classical data classification <https://link.springer.com/article/10.1007/s42484-021-00061-x>`__. 
"""
import pennylane as qml
import numpy as np

from typing import Union, List


def U_TTN(angle: Union[List, np.ndarray], wires: Union[List, np.ndarray]) -> None:
    """The quantum convolutional filter ansatz employing the architecture of
    `Quantum Tree Tensor Networks <https://iopscience.iop.org/article/10.1088/2058-9565/aaea94>`.
    The ansatz is parameterized with 2 rotation angles.

    Args:
        angle (Union[List, np.ndarray]): Rotation angles.
        wires (Union[List, np.ndarray]): The list of qubits on which the ansatz is
            applied.
    """
    qml.RY(angle[0], wires=wires[0])
    qml.RY(angle[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def U_RX(angle: Union[List, np.ndarray], wires: Union[List, np.ndarray]) -> None:
    """The quantum convolutional filter ansatz containing only :func:`pennylane.RX`
    gates.
    The ansatz is parameterized with 2 rotation angles.

    Args:
        angle (Union[List, np.ndarray]): Rotation angles.
        wires (Union[List, np.ndarray]): The list of qubits on which the ansatz is
            applied.
    """

    qml.RX(angle[0], wires=wires[0])
    qml.RX(angle[1], wires=wires[1])


def U_6(angle: Union[List, np.ndarray], wires: Union[List, np.ndarray]) -> None:
    # num_wires = 2
    # num_params = 10 #int: Number of trainable parameters that the operator depends on.

    qml.RX(angle[0], wires=wires[0])
    qml.RX(angle[1], wires=wires[1])
    qml.RZ(angle[2], wires=wires[0])
    qml.RZ(angle[3], wires=wires[1])
    qml.CRX(angle[4], wires=[wires[1], wires[0]])
    qml.CRX(angle[5], wires=[wires[0], wires[1]])
    qml.RX(angle[6], wires=wires[0])
    qml.RX(angle[7], wires=wires[1])
    qml.RZ(angle[8], wires=wires[0])
    qml.RZ(angle[9], wires=wires[1])


def U_SO4(angle: Union[List, np.ndarray], wires: Union[List, np.ndarray]) -> None:
    """The quantum convolutional filter ansatz generating an arbitrary :math:`SO_4`
    quantum state [1].
    The ansatz is parameterized with 6 rotation angles.

    Args:
        angle (Union[List, np.ndarray]): Rotation angles.
        wires (Union[List, np.ndarray]): The list of qubits on which the ansatz is
            applied.
    """

    qml.RY(angle[0], wires=wires[0])
    qml.RY(angle[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(angle[2], wires=wires[0])
    qml.RY(angle[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(angle[4], wires=wires[0])
    qml.RY(angle[5], wires=wires[1])


def U_SU4(angle: Union[List, np.ndarray], wires: Union[List, np.ndarray]) -> None:
    """The quantum convolutional filter ansatz generating an arbitrary :math:`SU_4`
    quantum state [1].
    The ansatz is parameterized with 15 rotation angles.

    Args:
        angle (Union[List, np.ndarray]): Rotation angles.
        wires (Union[List, np.ndarray]): The list of qubits on which the ansatz is
            applied.
    """

    qml.U3(angle[0], angle[1], angle[2], wires=wires[0])
    qml.U3(angle[3], angle[4], angle[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(angle[6], wires=wires[0])
    qml.RZ(angle[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(angle[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(angle[9], angle[10], angle[11], wires=wires[0])
    qml.U3(angle[12], angle[13], angle[14], wires=wires[1])


def Pooling_ansatz(
    angle: Union[List, np.ndarray], wires: Union[List, np.ndarray]
) -> None:
    # num_wires = 2
    # num_params = 2 #int: Number of trainable parameters that the operator depends on

    qml.CRZ(angle[0], wires=[wires[1], wires[0]])
    qml.PauliX(wires=wires[1])
    qml.CRX(angle[1], wires=[wires[1], wires[0]])


def U_ZZ(angle: Union[List, np.ndarray], wires: Union[List, np.ndarray]) -> None:
    # num_wires = 2
    # num_params = 7

    qml.RX(angle[0], wires=wires[0])
    qml.RZ(angle[1], wires=wires[0])
    qml.RX(angle[2], wires=wires[0])

    qml.RX(angle[3], wires=wires[1])
    qml.RZ(angle[4], wires=wires[1])
    qml.RX(angle[5], wires=wires[1])
    qml.IsingZZ(angle[6], wires=wires)


def U_qiskit(angle: Union[List, np.ndarray], wires: Union[List, np.ndarray]) -> None:
    # num_wires = 2
    # num_params = 3 #int: Number of trainable parameters that the operator depends on.

    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(angle[0], wires=wires[0])
    qml.RZ(angle[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

    qml.RY(angle[2], wires=wires[1])
