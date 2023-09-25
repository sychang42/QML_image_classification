import os
import sys

sys.path.append(os.path.dirname(__file__))

import pennylane as qml


# Convolutional layers
def conv_layer(U, params, wires):
    U(params, wires=[wires[0], wires[-1]])

    for i in range(0, len(wires), 2):
        U(params, wires=[wires[i], wires[i + 1]])

    for i in range(1, len(wires) - 1, 2):
        U(params, wires=[wires[i], wires[i + 1]])


# Pooling layers
def pooling_layer(V, params, wires):
    for i in range(0, len(wires), 2):
        V(params, wires=[wires[i + 1], wires[i]])
