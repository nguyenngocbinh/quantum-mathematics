# Ngày 13: Quantum Neural Networks cho Credit Scoring

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum neural networks và classical neural networks
- Nắm vững cách quantum neural networks cải thiện credit scoring
- Implement quantum neural networks cho credit risk assessment
- So sánh performance giữa quantum và classical neural networks

## 📚 Lý thuyết

### **Neural Networks Fundamentals**

#### **1. Classical Neural Networks**

**Feedforward Network:**
```
y = σ(Wₙσ(Wₙ₋₁...σ(W₁x + b₁)...) + bₙ)
```

**Backpropagation:**
```
∂L/∂W = ∂L/∂y × ∂y/∂W
```

**Activation Functions:**
- ReLU: `f(x) = max(0, x)`
- Sigmoid: `f(x) = 1/(1 + e⁻ˣ)`
- Tanh: `f(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)`

#### **2. Quantum Neural Networks**

**Quantum Circuit:**
```
|ψ⟩ = U(θ)|0⟩
```

**Parameterized Quantum Circuit:**
```
U(θ) = ∏ᵢ Uᵢ(θᵢ)
```

**Quantum Gradient:**
```
∂⟨ψ(θ)|O|ψ(θ)⟩/∂θ = (⟨ψ(θ + π/2)|O|ψ(θ + π/2)⟩ - ⟨ψ(θ - π/2)|O|ψ(θ - π/2)⟩)/2
```

### **Quantum Neural Network Types**

#### **1. Variational Quantum Circuits (VQC):**
```
|ψ⟩ = U(θ)U(x)|0⟩
```

#### **2. Quantum Neural Networks (QNN):**
```
y = ⟨ψ(x, θ)|O|ψ(x, θ)⟩
```

#### **3. Hybrid Quantum-Classical Networks:**
```
y = f_classical(⟨ψ(x, θ)|O|ψ(x, θ)⟩)
```

### **Quantum Neural Network Advantages**

#### **1. Quantum Properties:**
- **Superposition**: Parallel processing of multiple states
- **Entanglement**: Complex feature interactions
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Credit-specific Benefits:**
- **Non-linear Patterns**: Quantum circuits capture complex relationships
- **Feature Interactions**: Entanglement models credit correlations
- **Quantum Advantage**: Potential speedup for large datasets

## 💻 Thực hành

### **Project 13: Quantum Neural Networks cho Credit Scoring**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA, ADAM
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import CircuitQNN
import pennylane as qml

class ClassicalNeuralNetwork:
    """Classical neural network implementation"""
    
    def __init__(self, hidden_layer_sizes=(100, 50), max_iter=1000):
        self.nn = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """
        Prepare features for neural network
        """
        # Feature engineering
        features = data.copy()
        
        # Create interaction features
        features['debt_income_ratio'] = features['debt'] / (features['income'] + 1)
        features['credit_utilization'] = features['credit_used'] / (features['credit_limit'] + 1)
        features['payment_ratio'] = features['payments_made'] / (features['payments_due'] + 1)
        features['income_credit_ratio'] = features['income'] / (features['credit_limit'] + 1)
        
        # Normalize features
        numeric_features = features.select_dtypes(include=[np.number])
        if 'default' in numeric_features.columns:
            numeric_features = numeric_features.drop('default', axis=1)
        
        normalized_features = self.scaler.fit_transform(numeric_features)
        
        return pd.DataFrame(normalized_features, columns=numeric_features.columns)
    
    def train(self, X_train, y_train):
        """
        Train classical neural network
        """
        self.nn.fit(X_train, y_train)
        
    def predict(self, X_test):
        """
        Make predictions
        """
        return self.nn.predict(X_test)
    
    def predict_proba(self, X_test):
        """
        Predict probabilities
        """
        return self.nn.predict_proba(X_test)[:, 1]

class QuantumNeuralNetwork:
    """Quantum neural network implementation"""
    
    def __init__(self, num_qubits=4, num_layers=2):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.backend = Aer.get_backend('qasm_simulator')
        self.optimizer = SPSA(maxiter=100)
        self.parameters = None
        self.feature_map = None
        self.ansatz = None
        
    def create_feature_map(self, X):
        """
        Create quantum feature map
        """
        self.feature_map = ZZFeatureMap(
            feature_dimension=X.shape[1],
            reps=1
        )
        return self.feature_map
    
    def create_ansatz(self):
        """
        Create parameterized quantum circuit (ansatz)
        """
        self.ansatz = RealAmplitudes(
            num_qubits=self.num_qubits,
            reps=self.num_layers
        )
        return self.ansatz
    
    def create_quantum_circuit(self, X):
        """
        Create complete quantum circuit
        """
        if self.feature_map is None:
            self.create_feature_map(X)
        if self.ansatz is None:
            self.create_ansatz()
        
        # Combine feature map and ansatz
        circuit = self.feature_map.compose(self.ansatz)
        return circuit
    
    def quantum_forward(self, X, parameters):
        """
        Forward pass through quantum circuit
        """
        circuit = self.create_quantum_circuit(X)
        
        # Bind parameters
        bound_circuit = circuit.bind_parameters(parameters)
        
        # Execute circuit
        job = execute(bound_circuit, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate expectation value
        expectation = self._calculate_expectation(counts)
        
        return expectation
    
    def _calculate_expectation(self, counts):
        """
        Calculate expectation value from measurement counts
        """
        total_shots = sum(counts.values())
        expectation = 0.0
        
        for bitstring, count in counts.items():
            # Calculate parity (number of 1s)
            parity = sum(int(bit) for bit in bitstring) % 2
            probability = count / total_shots
            
            # Expectation value based on parity
            expectation += probability * (1 if parity == 0 else -1)
        
        return expectation
    
    def loss_function(self, parameters, X, y):
        """
        Loss function for quantum neural network
        """
        predictions = []
        
        for i in range(len(X)):
            # Encode input data
            input_params = np.concatenate([X[i], parameters])
            pred = self.quantum_forward(X[i:i+1], input_params)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Convert to probabilities
        probabilities = 1 / (1 + np.exp(-predictions))
        
        # Binary cross-entropy loss
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
        
        return loss
    
    def train(self, X_train, y_train):
        """
        Train quantum neural network
        """
        # Initialize parameters
        circuit = self.create_quantum_circuit(X_train)
        self.parameters = np.random.random(circuit.num_parameters) * 2 * np.pi
        
        # Optimize parameters
        result = self.optimizer.minimize(
            fun=lambda params: self.loss_function(params, X_train, y_train),
            x0=self.parameters
        )
        
        self.parameters = result.x
        print(f"Training completed. Final loss: {result.fun:.4f}")
    
    def predict(self, X_test):
        """
        Make predictions
        """
        predictions = []
        
        for i in range(len(X_test)):
            input_params = np.concatenate([X_test[i], self.parameters])
            pred = self.quantum_forward(X_test[i:i+1], input_params)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return (predictions > 0).astype(int)
    
    def predict_proba(self, X_test):
        """
        Predict probabilities
        """
        predictions = []
        
        for i in range(len(X_test)):
            input_params = np.concatenate([X_test[i], self.parameters])
            pred = self.quantum_forward(X_test[i:i+1], input_params)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        probabilities = 1 / (1 + np.exp(-predictions))
        return probabilities

class HybridQuantumNeuralNetwork:
    """Hybrid quantum-classical neural network"""
    
    def __init__(self, num_qubits=4, classical_layers=(50, 25)):
        self.num_qubits = num_qubits
        self.classical_layers = classical_layers
        self.backend = Aer.get_backend('qasm_simulator')
        self.quantum_parameters = None
        self.classical_weights = None
        self.classical_biases = None
        
    def create_quantum_layer(self):
        """
        Create quantum layer
        """
        # Feature map
        feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=1)
        
        # Ansatz
        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=2)
        
        # Combine
        quantum_circuit = feature_map.compose(ansatz)
        
        return quantum_circuit
    
    def quantum_forward(self, X, quantum_params):
        """
        Quantum forward pass
        """
        circuit = self.create_quantum_layer()
        
        quantum_features = []
        for i in range(len(X)):
            # Encode input
            input_params = np.concatenate([X[i], quantum_params])
            bound_circuit = circuit.bind_parameters(input_params)
            
            # Execute
            job = execute(bound_circuit, self.backend, shots=1000)
            result = job.result()
            counts = result.get_counts()
            
            # Extract features
            features = self._extract_quantum_features(counts)
            quantum_features.append(features)
        
        return np.array(quantum_features)
    
    def _extract_quantum_features(self, counts):
        """
        Extract features from quantum measurements
        """
        total_shots = sum(counts.values())
        features = []
        
        # Extract probabilities for each basis state
        for i in range(2**self.num_qubits):
            bitstring = format(i, f'0{self.num_qubits}b')
            count = counts.get(bitstring, 0)
            probability = count / total_shots
            features.append(probability)
        
        return features
    
    def classical_forward(self, X):
        """
        Classical forward pass
        """
        if self.classical_weights is None:
            return X
        
        current_input = X
        
        for i, (weights, bias) in enumerate(zip(self.classical_weights, self.classical_biases)):
            # Linear transformation
            linear_output = np.dot(current_input, weights) + bias
            
            # Activation function (ReLU)
            if i < len(self.classical_weights) - 1:  # Not the last layer
                current_input = np.maximum(0, linear_output)
            else:  # Last layer - sigmoid
                current_input = 1 / (1 + np.exp(-linear_output))
        
        return current_input
    
    def initialize_parameters(self, input_size):
        """
        Initialize network parameters
        """
        # Quantum parameters
        circuit = self.create_quantum_layer()
        self.quantum_parameters = np.random.random(circuit.num_parameters) * 2 * np.pi
        
        # Classical parameters
        layer_sizes = [input_size] + list(self.classical_layers) + [1]
        self.classical_weights = []
        self.classical_biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            weight_std = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            weights = np.random.normal(0, weight_std, (layer_sizes[i], layer_sizes[i + 1]))
            bias = np.zeros(layer_sizes[i + 1])
            
            self.classical_weights.append(weights)
            self.classical_biases.append(bias)
    
    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        """
        Train hybrid network
        """
        # Initialize parameters
        self.initialize_parameters(X_train.shape[1])
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            quantum_features = self.quantum_forward(X_train, self.quantum_parameters)
            predictions = self.classical_forward(quantum_features).flatten()
            
            # Calculate loss
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            loss = -np.mean(y_train * np.log(predictions) + (1 - y_train) * np.log(1 - predictions))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
            # Backward pass (simplified)
            # In practice, use proper backpropagation
            self._update_parameters(X_train, y_train, predictions, learning_rate)
    
    def _update_parameters(self, X_train, y_train, predictions, learning_rate):
        """
        Update parameters (simplified)
        """
        # Simplified parameter update
        # In practice, implement proper gradients
        
        # Update quantum parameters
        self.quantum_parameters += learning_rate * np.random.normal(0, 0.1, self.quantum_parameters.shape)
        
        # Update classical parameters (simplified)
        for i in range(len(self.classical_weights)):
            self.classical_weights[i] += learning_rate * np.random.normal(0, 0.1, self.classical_weights[i].shape)
            self.classical_biases[i] += learning_rate * np.random.normal(0, 0.1, self.classical_biases[i].shape)
    
    def predict(self, X_test):
        """
        Make predictions
        """
        quantum_features = self.quantum_forward(X_test, self.quantum_parameters)
        predictions = self.classical_forward(quantum_features).flatten()
        return (predictions > 0.5).astype(int)
    
    def predict_proba(self, X_test):
        """
        Predict probabilities
        """
        quantum_features = self.quantum_forward(X_test, self.quantum_parameters)
        predictions = self.classical_forward(quantum_features).flatten()
        return predictions

def generate_credit_data(n_samples=1000):
    """
    Generate synthetic credit data
    """
    np.random.seed(42)
    
    # Generate features
    income = np.random.normal(50000, 20000, n_samples)
    debt = np.random.uniform(10000, 100000, n_samples)
    credit_used = np.random.uniform(1000, 50000, n_samples)
    credit_limit = np.random.uniform(5000, 100000, n_samples)
    payments_made = np.random.uniform(0, 12, n_samples)
    payments_due = np.random.uniform(1, 12, n_samples)
    age = np.random.uniform(25, 65, n_samples)
    employment_years = np.random.uniform(0, 30, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'income': income,
        'debt': debt,
        'credit_used': credit_used,
        'credit_limit': credit_limit,
        'payments_made': payments_made,
        'payments_due': payments_due,
        'age': age,
        'employment_years': employment_years
    })
    
    # Create target variable
    debt_income_ratio = data['debt'] / (data['income'] + 1)
    credit_utilization = data['credit_used'] / (data['credit_limit'] + 1)
    payment_ratio = data['payments_made'] / (data['payments_due'] + 1)
    
    default_prob = (0.3 * debt_income_ratio + 
                   0.4 * credit_utilization + 
                   0.3 * (1 - payment_ratio))
    
    default_prob += np.random.normal(0, 0.1, n_samples)
    default_prob = np.clip(default_prob, 0, 1)
    
    data['default'] = (default_prob > 0.5).astype(int)
    
    return data

def compare_neural_networks():
    """
    Compare classical and quantum neural networks
    """
    print("=== Classical vs Quantum Neural Networks ===\n")
    
    # Generate data
    data = generate_credit_data(300)
    
    # Prepare features
    classical_nn = ClassicalNeuralNetwork(hidden_layer_sizes=(100, 50))
    features = classical_nn.prepare_features(data)
    
    # Split data
    X = features
    y = data['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Classical Neural Network
    print("1. Classical Neural Network:")
    classical_nn.train(X_train, y_train)
    classical_pred = classical_nn.predict(X_test)
    classical_proba = classical_nn.predict_proba(X_test)
    classical_auc = roc_auc_score(y_test, classical_proba)
    
    print(f"   AUC Score: {classical_auc:.4f}")
    print(f"   Accuracy: {(classical_pred == y_test).mean():.4f}")
    
    # Quantum Neural Network
    print("\n2. Quantum Neural Network:")
    quantum_nn = QuantumNeuralNetwork(num_qubits=4, num_layers=2)
    
    # Use subset of features for quantum NN
    quantum_features = X[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
        quantum_features, y, test_size=0.2, random_state=42
    )
    
    # Train quantum NN
    quantum_nn.train(X_train_q.values, y_train_q.values)
    quantum_pred = quantum_nn.predict(X_test_q.values)
    quantum_proba = quantum_nn.predict_proba(X_test_q.values)
    quantum_auc = roc_auc_score(y_test_q, quantum_proba)
    
    print(f"   AUC Score: {quantum_auc:.4f}")
    print(f"   Accuracy: {(quantum_pred == y_test_q).mean():.4f}")
    
    # Hybrid Neural Network
    print("\n3. Hybrid Quantum-Classical Neural Network:")
    hybrid_nn = HybridQuantumNeuralNetwork(num_qubits=4, classical_layers=(50, 25))
    
    # Train hybrid NN
    hybrid_nn.train(X_train_q.values, y_train_q.values, epochs=50)
    hybrid_pred = hybrid_nn.predict(X_test_q.values)
    hybrid_proba = hybrid_nn.predict_proba(X_test_q.values)
    hybrid_auc = roc_auc_score(y_test_q, hybrid_proba)
    
    print(f"   AUC Score: {hybrid_auc:.4f}")
    print(f"   Accuracy: {(hybrid_pred == y_test_q).mean():.4f}")
    
    # Compare results
    print(f"\n4. Comparison:")
    print(f"   Classical AUC: {classical_auc:.4f}")
    print(f"   Quantum AUC: {quantum_auc:.4f}")
    print(f"   Hybrid AUC: {hybrid_auc:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # ROC curves
    plt.subplot(1, 3, 1)
    fpr_classical, tpr_classical, _ = roc_curve(y_test, classical_proba)
    fpr_quantum, tpr_quantum, _ = roc_curve(y_test_q, quantum_proba)
    fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test_q, hybrid_proba)
    
    plt.plot(fpr_classical, tpr_classical, label=f'Classical NN (AUC = {classical_auc:.3f})')
    plt.plot(fpr_quantum, tpr_quantum, label=f'Quantum NN (AUC = {quantum_auc:.3f})')
    plt.plot(fpr_hybrid, tpr_hybrid, label=f'Hybrid NN (AUC = {hybrid_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Neural Network Comparison')
    plt.legend()
    plt.grid(True)
    
    # AUC comparison
    plt.subplot(1, 3, 2)
    methods = ['Classical NN', 'Quantum NN', 'Hybrid NN']
    auc_scores = [classical_auc, quantum_auc, hybrid_auc]
    plt.bar(methods, auc_scores, color=['blue', 'orange', 'green'])
    plt.ylabel('AUC Score')
    plt.title('AUC Score Comparison')
    plt.ylim(0, 1)
    
    # Accuracy comparison
    plt.subplot(1, 3, 3)
    accuracies = [
        (classical_pred == y_test).mean(),
        (quantum_pred == y_test_q).mean(),
        (hybrid_pred == y_test_q).mean()
    ]
    plt.bar(methods, accuracies, color=['blue', 'orange', 'green'])
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return (classical_auc, quantum_auc, hybrid_auc, 
            classical_proba, quantum_proba, hybrid_proba)

def quantum_neural_network_analysis():
    """
    Analyze quantum neural network properties
    """
    print("=== Quantum Neural Network Analysis ===\n")
    
    # Generate data
    data = generate_credit_data(200)
    features = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Create quantum neural network
    qnn = QuantumNeuralNetwork(num_qubits=4, num_layers=2)
    
    # Analyze circuit properties
    circuit = qnn.create_quantum_circuit(normalized_features)
    print(f"1. Circuit Analysis:")
    print(f"   Number of qubits: {circuit.num_qubits}")
    print(f"   Circuit depth: {circuit.depth()}")
    print(f"   Number of parameters: {circuit.num_parameters}")
    print(f"   Number of gates: {circuit.count_ops()}")
    
    # Analyze parameter sensitivity
    print(f"\n2. Parameter Sensitivity Analysis:")
    base_params = np.random.random(circuit.num_parameters) * 2 * np.pi
    
    sensitivities = []
    for i in range(circuit.num_parameters):
        # Perturb parameter
        perturbed_params = base_params.copy()
        perturbed_params[i] += 0.1
        
        # Calculate output difference
        base_output = qnn.quantum_forward(normalized_features[:1], base_params)
        perturbed_output = qnn.quantum_forward(normalized_features[:1], perturbed_params)
        
        sensitivity = abs(perturbed_output - base_output)
        sensitivities.append(sensitivity)
    
    print(f"   Average sensitivity: {np.mean(sensitivities):.4f}")
    print(f"   Max sensitivity: {np.max(sensitivities):.4f}")
    print(f"   Min sensitivity: {np.min(sensitivities):.4f}")
    
    # Analyze training dynamics
    print(f"\n3. Training Dynamics Analysis:")
    
    # Simulate training steps
    training_losses = []
    test_accuracies = []
    
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, data['default'], test_size=0.2, random_state=42
    )
    
    # Initialize parameters
    params = np.random.random(circuit.num_parameters) * 2 * np.pi
    
    for step in range(20):
        # Calculate loss
        loss = qnn.loss_function(params, X_train, y_train)
        training_losses.append(loss)
        
        # Calculate test accuracy
        predictions = qnn.predict(X_test)
        accuracy = (predictions == y_test).mean()
        test_accuracies.append(accuracy)
        
        # Update parameters (simplified)
        params += 0.1 * np.random.normal(0, 0.1, params.shape)
        
        if step % 5 == 0:
            print(f"   Step {step}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    # Plot training dynamics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, 'b-', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, 'r-', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return training_losses, test_accuracies

# Exercise: Quantum Neural Network Architecture Search
def quantum_neural_network_architecture_search():
    """
    Exercise: Search for optimal quantum neural network architecture
    """
    print("=== Quantum Neural Network Architecture Search ===\n")
    
    # Generate data
    data = generate_credit_data(300)
    features = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, data['default'], test_size=0.2, random_state=42
    )
    
    # Test different architectures
    architectures = [
        {'num_qubits': 2, 'num_layers': 1},
        {'num_qubits': 2, 'num_layers': 2},
        {'num_qubits': 4, 'num_layers': 1},
        {'num_qubits': 4, 'num_layers': 2},
        {'num_qubits': 4, 'num_layers': 3},
        {'num_qubits': 6, 'num_layers': 1},
        {'num_qubits': 6, 'num_layers': 2}
    ]
    
    results = {}
    
    for i, arch in enumerate(architectures):
        print(f"Testing Architecture {i+1}: {arch}")
        
        try:
            # Create quantum neural network
            qnn = QuantumNeuralNetwork(
                num_qubits=arch['num_qubits'],
                num_layers=arch['num_layers']
            )
            
            # Train network
            qnn.train(X_train, y_train)
            
            # Evaluate
            predictions = qnn.predict(X_test)
            probabilities = qnn.predict_proba(X_test)
            accuracy = (predictions == y_test).mean()
            auc = roc_auc_score(y_test, probabilities)
            
            results[f"Arch_{i+1}"] = {
                'architecture': arch,
                'accuracy': accuracy,
                'auc': auc,
                'num_parameters': qnn.create_quantum_circuit(X_train).num_parameters
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  Parameters: {qnn.create_quantum_circuit(X_train).num_parameters}")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # AUC comparison
    plt.subplot(1, 3, 1)
    arch_names = list(results.keys())
    auc_scores = [results[name]['auc'] for name in arch_names]
    plt.bar(arch_names, auc_scores, color='skyblue')
    plt.ylabel('AUC Score')
    plt.title('AUC Score by Architecture')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Accuracy comparison
    plt.subplot(1, 3, 2)
    accuracies = [results[name]['accuracy'] for name in arch_names]
    plt.bar(arch_names, accuracies, color='lightcoral')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Architecture')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Parameters vs Performance
    plt.subplot(1, 3, 3)
    parameters = [results[name]['num_parameters'] for name in arch_names]
    plt.scatter(parameters, auc_scores, s=100, alpha=0.7)
    plt.xlabel('Number of Parameters')
    plt.ylabel('AUC Score')
    plt.title('Parameters vs Performance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run demos
if __name__ == "__main__":
    print("Running Neural Network Comparisons...")
    classical_auc, quantum_auc, hybrid_auc, classical_proba, quantum_proba, hybrid_proba = compare_neural_networks()
    
    print("\nRunning Quantum Neural Network Analysis...")
    training_losses, test_accuracies = quantum_neural_network_analysis()
    
    print("\nRunning Architecture Search...")
    architecture_results = quantum_neural_network_architecture_search()

### **Exercise 2: Quantum Neural Network Optimization**

```python
def quantum_neural_network_optimization():
    """
    Exercise: Optimize quantum neural network parameters
    """
    from scipy.optimize import minimize
    
    def objective_function(params):
        """
        Objective function for quantum neural network optimization
        """
        # Generate data
        data = generate_credit_data(100)
        features = data[['income', 'debt', 'credit_used', 'credit_limit']].copy()
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            normalized_features, data['default'], test_size=0.2, random_state=42
        )
        
        # Create quantum neural network
        qnn = QuantumNeuralNetwork(num_qubits=4, num_layers=2)
        
        # Set parameters
        qnn.parameters = params
        
        # Calculate loss
        loss = qnn.loss_function(params, X_train, y_train)
        
        return loss
    
    # Optimize parameters
    initial_params = np.random.random(20) * 2 * np.pi  # 20 parameters
    
    result = minimize(objective_function, initial_params, method='L-BFGS-B')
    
    print("=== Quantum Neural Network Optimization ===")
    print(f"Optimization Success: {result.success}")
    print(f"Final Loss: {result.fun:.4f}")
    print(f"Number of Iterations: {result.nit}")
    
    return result

# Run optimization
if __name__ == "__main__":
    opt_result = quantum_neural_network_optimization()
```

## 📊 Kết quả và Phân tích

### **Quantum Neural Network Advantages:**

#### **1. Quantum Properties:**
- **Superposition**: Parallel processing of multiple states
- **Entanglement**: Complex feature interactions
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Credit-specific Benefits:**
- **Non-linear Patterns**: Quantum circuits capture complex relationships
- **Feature Interactions**: Entanglement models credit correlations
- **Quantum Advantage**: Potential speedup for large datasets

#### **3. Performance Characteristics:**
- **Better Convergence**: Quantum gradients improve training
- **Robustness**: Quantum circuits handle noisy credit data
- **Scalability**: Quantum advantage for large-scale credit scoring

### **Comparison với Classical Neural Networks:**

#### **Classical Limitations:**
- Limited non-linear transformations
- Gradient vanishing/exploding
- Local minima problems
- Feature engineering required

#### **Quantum Advantages:**
- Rich quantum feature space
- Quantum gradient methods
- Global optimization potential
- Automatic feature learning

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Neural Network Calibration**
Implement quantum neural network calibration methods cho credit scoring.

### **Exercise 2: Quantum Neural Network Ensemble Methods**
Build ensemble of quantum neural networks cho improved performance.

### **Exercise 3: Quantum Neural Network Feature Selection**
Develop quantum feature selection cho neural network optimization.

### **Exercise 4: Quantum Neural Network Validation**
Create validation framework cho quantum neural network models.

---

> *"Quantum Neural Networks leverage quantum superposition and entanglement to provide superior learning capabilities for complex credit risk patterns."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Clustering cho Customer Segmentation](Day14.md) 