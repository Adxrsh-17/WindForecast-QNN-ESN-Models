import pennylane as qml
from pennylane.qnn import KerasLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
import numpy as np

# Set number of qubits
nq = 4
dev = qml.device("default.qubit", wires=nq)

# Define quantum circuit
@qml.qnode(dev)
def qc(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(nq), rotation='Y')
    qml.templates.StronglyEntanglingLayers(weights, wires=range(nq))
    return [qml.expval(qml.PauliZ(i)) for i in range(nq)]

# Define weight shapes
weight_shapes = {"weights": (3, nq, 3)}  # 3 layers of entangling blocks

# Wrap in Keras layer
qlayer = KerasLayer(qc, weight_shapes, output_dim=nq)

# Classical model
model = Sequential([
    Flatten(input_shape=(nq,)),
    Dense(nq),
    qlayer,
    Dense(1, activation="linear")
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Dummy training data
X = np.random.rand(100, nq)
y = np.random.rand(100, 1)

# Train
model.fit(X, y, epochs=5, batch_size=10)
