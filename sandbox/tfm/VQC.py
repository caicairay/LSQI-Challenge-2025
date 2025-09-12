import pennylane as qml
import torch
import torch.nn as nn

# from catalyst import qjit


class VQC_class(torch.nn.Module):
    def __init__(self, input_size, output_size, n_layers=1, n_qubits=4) -> None:
        super().__init__()
        self.sim_dev = qml.device("default.qubit", wires=n_qubits)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.clayer_1 = torch.nn.Linear(input_size, n_qubits)
        # self.qlayer = qml.qnn.TorchLayer(qnode, weights_shape)
        self.clayer_2 = torch.nn.Linear(n_qubits, output_size)
        # weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        weights_shape = (n_layers, n_qubits)
        self.weights = nn.Parameter(torch.randn(weights_shape))

    def QNode(self, inputs, weights):
        @qml.qnode(self.sim_dev, interface="torch", diff_method="backprop")
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        has_batch_dim = len(inputs.shape) > 1

        # in case the input has more than one batch dimension
        if has_batch_dim:
            batch_dims = inputs.shape[:-1]
            inputs = torch.reshape(inputs, (-1, inputs.shape[-1]))

        # calculate the forward pass as usual
        weights.to(inputs.device)
        res = qnode(inputs, weights)
        if isinstance(res, torch.Tensor):
            results = res.type(inputs.dtype)

        else:
            if len(inputs.shape) > 1:
                res = [torch.reshape(r, (inputs.shape[0], -1)) for r in res]
            results = torch.hstack(res).type(inputs.dtype)

        # reshape to the correct number of batch dims
        if has_batch_dim:
            results = torch.reshape(results, (*batch_dims, *results.shape[1:]))

        return results

    def forward(self, X):
        X = self.clayer_1(X)
        X = self.QNode(X, self.weights)
        X = self.clayer_2(X)
        return X

    def _freeze_Qnode(self):
        self.weights.requires_grad = False

    def _unfreeze_Qnode(self):
        self.weights.requires_grad = True


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import make_moons

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    X, y = make_moons(n_samples=200, noise=0.1)
    y_ = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
    y_hot = torch.scatter(torch.zeros((200, 2)), 1, y_, 1)

    # c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in y]  # colours for each class
    # plt.axis("off")
    # plt.scatter(X[:, 0], X[:, 1], c=c)
    # plt.show()

    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        # qml.Hadamard(wires=[0]),
        # qml.Hadamard(wires=[1]),
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        # qml.AngleEmbedding(inputs**2, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        # qml.Rot(phi, theta, omega, wires=list(range(n_qubits)))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    # @qml.qnode(dev)
    # def qnode(inputs, weights):
    #     qml.AngleEmbedding(inputs, wires=range(n_qubits))
    #     qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.Rot)
    #     return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    n_layers = 1
    weight_shapes = {
        "weights": (n_layers, n_qubits),
    }
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    clayer_1 = torch.nn.Linear(2, 2)
    clayer_2 = torch.nn.Linear(2, 2)
    softmax = torch.nn.Softmax(dim=1)
    layers = [clayer_1, qlayer, clayer_2, softmax]
    model = torch.nn.Sequential(*layers)

    opt = torch.optim.SGD(model.parameters(), lr=0.2)
    loss = torch.nn.L1Loss()

    X = torch.tensor(X, requires_grad=True).float()
    y_hot = y_hot.float()

    batch_size = 5
    batches = 200 // batch_size

    data_loader = torch.utils.data.DataLoader(
        list(zip(X, y_hot)), batch_size=5, shuffle=True, drop_last=True
    )

    epochs = 6

    for epoch in range(epochs):

        running_loss = 0

        for xs, ys in data_loader:
            opt.zero_grad()

            loss_evaluated = loss(model(xs), ys)
            loss_evaluated.backward()

            opt.step()

            running_loss += loss_evaluated

        avg_loss = running_loss / batches
        print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

    y_pred = model(X)
    predictions = torch.argmax(y_pred, axis=1).detach().numpy()

    correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y)]
    accuracy = sum(correct) / len(correct)
    print(f"Accuracy: {accuracy * 100}%")
