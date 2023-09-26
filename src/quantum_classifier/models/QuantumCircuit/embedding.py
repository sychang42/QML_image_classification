# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.templates.state_preparations import MottonenStatePreparation
import numpy as np


def get_data_embedding(embedding_type="Angular-Hybrid4-2"):
    # Hybrid Angle Embedding (HAE)

    if embedding_type == "Angular":

        def circuit(X):
            AngleEmbedding(X[: len(X) // 2], wires=range(len(X) // 2), rotation="X")
            AngleEmbedding(X[len(X) // 2 :], wires=range(len(X) // 2), rotation="Y")

    return circuit
