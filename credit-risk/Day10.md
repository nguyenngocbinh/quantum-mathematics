# Ngày 10: Quantum Machine Learning Basics cho Finance

## 🎯 Mục tiêu học tập

- Hiểu fundamentals của quantum machine learning
- Implement quantum feature maps cho financial data
- Xây dựng quantum neural networks cho credit risk
- So sánh quantum ML với classical ML cho finance applications

## 📚 Lý thuyết

### **Quantum Machine Learning Fundamentals**

#### **1. Quantum Feature Maps**
Quantum feature maps encode classical data vào quantum states:

**Mathematical Foundation:**
```
φ(x) = U(x)|0⟩^⊗n
```

Trong đó:
- φ(x): Quantum feature map
- U(x): Parameterized quantum circuit
- |0⟩^⊗n: Initial quantum state

#### **2. Quantum Kernels**
Quantum kernels leverage quantum feature maps:

**Kernel Function:**
```
K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²
```

#### **3. Quantum Neural Networks**
Parameterized quantum circuits cho supervised learning:

**Cost Function:**
```
C(θ) = Σᵢ L(f(x_i; θ), y_i)
```

### **Quantum ML Advantages cho Finance**

#### **1. High-Dimensional Feature Spaces:**
- Exponential feature space với linear qubit growth
- Non-linear feature interactions
- Quantum entanglement cho complex correlations

#### **2. Quantum Speedup Potential:**
- Quantum kernel estimation
- Quantum gradient computation
- Quantum optimization algorithms

#### **3. Financial Applications:**
- Credit scoring với quantum features
- Portfolio optimization
- Risk factor modeling
- Market prediction

## 💻 Thực hành

### **Project 10: Quantum Machine Learning Framework cho Finance**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
import pennylane as qml

class QuantumFeatureMap:
    """Quantum feature map implementation"""
    
    def __init__(self, feature_dimension, reps=2):
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_feature_map(self, x):
        """
        Create quantum feature map circuit
        """
        # Use ZZFeatureMap for encoding
        feature_map = ZZFeatureMap(
            feature_dimension=self.feature_dimension,
            reps=self.reps,
            insert_barriers=True
        )
        
        # Bind parameters
        circuit = feature_map.bind_parameters(x)
        return circuit
    
    def compute_kernel_matrix(self, X1, X2=None):
        """
        Compute quantum kernel matrix
        """
        if X2 is None:
            X2 = X1
            
        n1 = len(X1)
        n2 = len(X2)
        kernel_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                # Create circuits for both data points
                circuit1 = self.create_feature_map(X1[i])
                circuit2 = self.create_feature_map(X2[j])
                
                # Compute overlap
                overlap = self._compute_overlap(circuit1, circuit2)
                kernel_matrix[i, j] = overlap
                
        return kernel_matrix
    
    def _compute_overlap(self, circuit1, circuit2):
        """
        Compute overlap between two quantum states
        """
        # Create combined circuit
        combined_circuit = circuit1.compose(circuit2.inverse())
        
        # Add measurement
        combined_circuit.measure_all()
        
        # Execute circuit
        job = execute(combined_circuit, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate overlap (probability of measuring |0⟩^⊗n)
        zero_state = '0' * combined_circuit.num_qubits
        overlap = counts.get(zero_state, 0) / 1000
        
        return overlap

class QuantumNeuralNetwork:
    """Quantum Neural Network for classification"""
    
    def __init__(self, feature_dimension, num_classes=2):
        self.feature_dimension = feature_dimension
        self.num_classes = num_classes
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_variational_circuit(self, x, params):
        """
        Create variational quantum circuit
        """
        # Feature map
        feature_map = ZZFeatureMap(
            feature_dimension=self.feature_dimension,
            reps=1
        )
        
        # Variational form
        var_form = RealAmplitudes(
            num_qubits=self.feature_dimension,
            reps=2
        )
        
        # Combine circuits
        circuit = feature_map.compose(var_form)
        
        # Bind parameters
        circuit = circuit.bind_parameters(np.concatenate([x, params]))
        
        return circuit
    
    def predict(self, x, params):
        """
        Make prediction using quantum circuit
        """
        circuit = self.create_variational_circuit(x, params)
        circuit.measure_all()
        
        # Execute circuit
        job = execute(circuit, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate prediction probability
        prediction = self._extract_prediction(counts)
        return prediction
    
    def _extract_prediction(self, counts):
        """
        Extract prediction from measurement counts
        """
        total_shots = sum(counts.values())
        
        # Simple heuristic: count states with even number of 1s
        even_count = 0
        for state, count in counts.items():
            if state.count('1') % 2 == 0:
                even_count += count
                
        return even_count / total_shots

def generate_financial_data(n_samples=200):
    """
    Generate synthetic financial data for credit risk
    """
    np.random.seed(42)
    
    # Generate features
    income = np.random.normal(50000, 20000, n_samples)
    debt_ratio = np.random.uniform(0.1, 0.8, n_samples)
    payment_history = np.random.uniform(0.5, 1.0, n_samples)
    credit_utilization = np.random.uniform(0.1, 0.9, n_samples)
    age = np.random.uniform(25, 65, n_samples)
    
    # Create features matrix
    X = np.column_stack([
        income, debt_ratio, payment_history, credit_utilization, age
    ])
    
    # Generate labels (0: good credit, 1: bad credit)
    # Simple rule: high debt ratio + low payment history = bad credit
    y = ((debt_ratio > 0.6) & (payment_history < 0.7)).astype(int)
    
    # Add some noise
    noise = np.random.random(n_samples) < 0.1
    y = (y + noise) % 2
    
    return X, y

def quantum_ml_demo():
    """
    Demo quantum machine learning cho credit risk
    """
    # Generate data
    X, y = generate_financial_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data Summary:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # 1. Quantum Feature Map Analysis
    print("\n1. Quantum Feature Map Analysis:")
    feature_map = QuantumFeatureMap(feature_dimension=X_train.shape[1])
    
    # Compute kernel matrix for training data
    kernel_matrix = feature_map.compute_kernel_matrix(X_train_scaled[:10, :])
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(kernel_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Quantum Kernel Matrix')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    
    # 2. Quantum Neural Network
    print("\n2. Quantum Neural Network Training:")
    qnn = QuantumNeuralNetwork(feature_dimension=X_train.shape[1])
    
    # Initialize parameters
    num_params = 2 * X_train.shape[1] * 2  # RealAmplitudes parameters
    params = np.random.random(num_params) * 2 * np.pi
    
    # Simple training loop
    predictions = []
    for x in X_train_scaled[:20]:  # Use subset for demo
        pred = qnn.predict(x, params)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_train[:20], binary_predictions)
    print(f"Training Accuracy: {accuracy:.4f}")
    
    # 3. Comparison with Classical ML
    print("\n3. Classical ML Comparison:")
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    # SVM
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    # Visualization
    plt.subplot(1, 3, 2)
    plt.scatter(X_train_scaled[:20, 0], X_train_scaled[:20, 1], 
               c=binary_predictions, cmap='viridis', alpha=0.7)
    plt.xlabel('Income (scaled)')
    plt.ylabel('Debt Ratio (scaled)')
    plt.title('Quantum NN Predictions')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    methods = ['Quantum NN', 'Logistic Regression', 'SVM']
    accuracies = [accuracy, lr_accuracy, svm_accuracy]
    colors = ['red', 'blue', 'green']
    
    bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'quantum_accuracy': accuracy,
        'lr_accuracy': lr_accuracy,
        'svm_accuracy': svm_accuracy,
        'kernel_matrix': kernel_matrix
    }

# Run demo
results = quantum_ml_demo()
```

## 📊 Bài tập về nhà

### **Bài tập 1: Quantum Feature Engineering**
- Implement custom quantum feature maps cho financial data
- Compare different quantum encoding strategies
- Analyze feature space dimensionality

### **Bài tập 2: Quantum Model Optimization**
- Optimize quantum neural network parameters
- Implement quantum gradient descent
- Compare optimization algorithms

### **Bài tập 3: Real Financial Data Application**
- Apply quantum ML to real credit risk dataset
- Implement cross-validation cho quantum models
- Document model performance và limitations

## 🔗 Tài liệu tham khảo

### **Papers:**
- "Quantum Machine Learning" - Schuld, Petruccione
- "Quantum Feature Maps and Kernels" - Various authors
- "Quantum Neural Networks" - Research papers

### **Books:**
- "Quantum Machine Learning" - Schuld, Petruccione
- "Quantum Computing for Finance" - Orús, Mugel, Lizaso

### **Online Resources:**
- [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/)
- [PennyLane Tutorials](https://pennylane.ai/qml/tutorials/)
- [Quantum ML Research Papers](https://quantum-journal.org/)

## 🎯 Kết luận

Ngày 10 đã cover:
- ✅ Quantum machine learning fundamentals
- ✅ Quantum feature maps và kernels
- ✅ Quantum neural networks cho finance
- ✅ Advanced quantum ML applications

**Chuẩn bị cho Phase 2**: Quantum Algorithms cho Credit Risk (Ngày 11-20).

---

> *"Quantum machine learning represents a paradigm shift in how we approach complex financial modeling problems."* - Quantum ML Research 