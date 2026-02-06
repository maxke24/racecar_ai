import numpy as np
import json
import os


class NeuralNetwork:
    """Simple feedforward neural network for car control."""

    def __init__(self, layer_sizes=None):
        # Default: 7 ray inputs + speed -> hidden 10 -> 2 outputs (throttle, steering)
        if layer_sizes is None:
            layer_sizes = [8, 10, 2]
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # Initialize weights with Xavier initialization
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, inputs):
        """Forward pass through the network."""
        x = np.array(inputs, dtype=np.float64)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                # ReLU for hidden layers
                x = np.maximum(0, x)
            else:
                # Tanh for output layer (bounded -1 to 1)
                x = np.tanh(x)
        return x

    def mutate(self, rate=0.2, strength=0.3):
        """Mutate weights and biases for genetic algorithm."""
        child = NeuralNetwork(self.layer_sizes)
        for i in range(len(self.weights)):
            child.weights[i] = self.weights[i].copy()
            child.biases[i] = self.biases[i].copy()

            # Random mutations
            mask = np.random.random(child.weights[i].shape) < rate
            child.weights[i] += mask * np.random.randn(*child.weights[i].shape) * strength

            mask = np.random.random(child.biases[i].shape) < rate
            child.biases[i] += mask * np.random.randn(*child.biases[i].shape) * strength

        return child

    def crossover(self, other):
        """Create a child by combining two networks."""
        child = NeuralNetwork(self.layer_sizes)
        for i in range(len(self.weights)):
            mask = np.random.random(self.weights[i].shape) > 0.5
            child.weights[i] = np.where(mask, self.weights[i], other.weights[i])
            mask = np.random.random(self.biases[i].shape) > 0.5
            child.biases[i] = np.where(mask, self.biases[i], other.biases[i])
        return child

    def copy(self):
        clone = NeuralNetwork(self.layer_sizes)
        for i in range(len(self.weights)):
            clone.weights[i] = self.weights[i].copy()
            clone.biases[i] = self.biases[i].copy()
        return clone

    def save(self, filepath):
        data = {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(filepath):
        with open(filepath) as f:
            data = json.load(f)
        nn = NeuralNetwork(data["layer_sizes"])
        nn.weights = [np.array(w) for w in data["weights"]]
        nn.biases = [np.array(b) for b in data["biases"]]
        return nn


class Population:
    """Manages a population of cars with neural networks for genetic algorithm training."""

    def __init__(self, size, nn_layer_sizes=None):
        self.size = size
        self.nn_layer_sizes = nn_layer_sizes or [8, 10, 2]
        self.networks = [NeuralNetwork(self.nn_layer_sizes) for _ in range(size)]
        self.generation = 0
        self.best_fitness = 0
        self.best_network = None

    def get_actions(self, car_index, ray_distances_normalized, speed_normalized):
        """Get throttle and steering from the neural network for a given car."""
        inputs = list(ray_distances_normalized) + [speed_normalized]
        outputs = self.networks[car_index].forward(inputs)
        throttle = (float(outputs[0]) + 1.0) / 2.0  # Remap from [-1,1] to [0,1] (forward only)
        steering = float(outputs[1])
        return throttle, steering

    def evolve(self, fitnesses):
        """Create the next generation using selection, crossover, and mutation."""
        self.generation += 1

        # Track best
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_network = self.networks[best_idx].copy()

        # Selection: tournament selection
        new_networks = []

        # Elitism: keep top 2
        sorted_indices = np.argsort(fitnesses)[::-1]
        for i in range(min(2, self.size)):
            new_networks.append(self.networks[sorted_indices[i]].copy())

        # Fill rest with crossover + mutation
        while len(new_networks) < self.size:
            # Tournament selection
            parent1 = self._tournament_select(fitnesses)
            parent2 = self._tournament_select(fitnesses)
            child = self.networks[parent1].crossover(self.networks[parent2])
            child = child.mutate(rate=0.2, strength=0.3)
            new_networks.append(child)

        self.networks = new_networks

    def _tournament_select(self, fitnesses, k=3):
        """Select the best individual from k random candidates."""
        candidates = np.random.choice(len(fitnesses), size=min(k, len(fitnesses)), replace=False)
        best = candidates[np.argmax([fitnesses[c] for c in candidates])]
        return best

    def save_best(self, filepath):
        if self.best_network:
            self.best_network.save(filepath)

    def load_best(self, filepath):
        if os.path.exists(filepath):
            self.best_network = NeuralNetwork.load(filepath)
            # Seed population with mutations of the best
            self.networks[0] = self.best_network.copy()
            for i in range(1, self.size):
                self.networks[i] = self.best_network.mutate(rate=0.3, strength=0.4)
            return True
        return False
