# Ngày 28: Quantum Credit Fraud Detection

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum credit fraud detection và classical fraud detection
- Nắm vững cách quantum computing cải thiện fraud detection
- Implement quantum credit fraud detection algorithms
- So sánh performance giữa quantum và classical fraud detection

## 📚 Lý thuyết

### **Credit Fraud Detection Fundamentals**

#### **1. Classical Fraud Detection**

**Detection Models:**
- **Logistic Regression**: Linear fraud detection
- **Random Forest**: Ensemble fraud detection
- **Neural Networks**: Deep learning for fraud
- **Anomaly Detection**: Outlier-based detection

**Fraud Metrics:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
AUC = Area under ROC curve
```

#### **2. Quantum Fraud Detection**

**Quantum State Encoding:**
```
|ψ⟩ = Σᵢ αᵢ|featureᵢ⟩
```

**Quantum Fraud Operator:**
```
H_fraud = Σᵢ Weightᵢ × Featureᵢ × |fraudᵢ⟩⟨fraudᵢ|
```

**Quantum Classification:**
```
Fraud_quantum = argmax(⟨ψ|H_fraud|ψ⟩)
```

### **Quantum Fraud Detection Methods**

#### **1. Quantum Feature Engineering:**
- **Quantum Feature Maps**: Encode features as quantum states
- **Quantum Kernel Methods**: Quantum kernel for fraud detection
- **Quantum Anomaly Detection**: Quantum anomaly detection

#### **2. Quantum Classification:**
- **Quantum SVM**: Quantum support vector machines
- **Quantum Neural Networks**: Quantum neural network models
- **Quantum Ensemble Methods**: Quantum ensemble fraud detection

#### **3. Quantum Fraud Calibration:**
- **Quantum Probability Estimation**: Quantum probability calculation
- **Quantum Confidence Intervals**: Quantum uncertainty quantification
- **Quantum Fraud Stability**: Quantum fraud stability analysis

## 💻 Thực hành

### **Project 28: Quantum Credit Fraud Detection Framework**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA, COBYLA

class ClassicalFraudDetection:
    """Classical fraud detection models"""
    def __init__(self):
        self.scaler = StandardScaler()
    def generate_fraud_data(self, n_samples=1000):
        np.random.seed(42)
        # Features
        amount = np.random.exponential(200, n_samples)
        time = np.random.uniform(0, 24, n_samples)
        location = np.random.randint(0, 10, n_samples)
        account_age = np.random.exponential(5, n_samples)
        # Fraud label (imbalanced)
        fraud_prob = 0.02 + 0.1 * (amount > 500) + 0.05 * (account_age < 1)
        fraud = np.random.binomial(1, np.clip(fraud_prob, 0, 1))
        data = pd.DataFrame({
            'amount': amount,
            'time': time,
            'location': location,
            'account_age': account_age
        })
        return data, fraud
    def logistic_regression_fraud(self, X_train, y_train, X_test):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        return predictions, probabilities
class QuantumFraudDetection:
    """Quantum fraud detection implementation"""
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.optimizer = SPSA(maxiter=100)
    def create_fraud_circuit(self, features):
        feature_map = ZZFeatureMap(feature_dimension=len(features), reps=2)
        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=3)
        circuit = feature_map.compose(ansatz)
        return circuit
    def quantum_fraud_prediction(self, X_train, y_train, X_test):
        predictions = []
        probabilities = []
        for i, sample in enumerate(X_test):
            if i >= 50:
                break
            circuit = self.create_fraud_circuit(sample)
            job = execute(circuit, self.backend, shots=1000)
            result = job.result()
            counts = result.get_counts()
            prediction, probability = self._extract_prediction_from_counts(counts)
            predictions.append(prediction)
            probabilities.append(probability)
        return np.array(predictions), np.array(probabilities)
    def _extract_prediction_from_counts(self, counts):
        total_shots = sum(counts.values())
        fraud_prob = 0.0
        for bitstring, count in counts.items():
            probability = count / total_shots
            parity = sum(int(bit) for bit in bitstring) % 2
            if parity == 1:
                fraud_prob += probability
        prediction = 1 if fraud_prob > 0.5 else 0
        return prediction, fraud_prob

def compare_fraud_detection():
    print("=== Classical vs Quantum Fraud Detection ===\n")
    classical_detector = ClassicalFraudDetection()
    data, labels = classical_detector.generate_fraud_data(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    X_train_scaled = classical_detector.scaler.fit_transform(X_train)
    X_test_scaled = classical_detector.scaler.transform(X_test)
    # Classical
    print("1. Classical Fraud Detection:")
    classical_predictions, classical_probabilities = classical_detector.logistic_regression_fraud(X_train_scaled, y_train, X_test_scaled)
    print(classification_report(y_test, classical_predictions))
    auc_classical = roc_auc_score(y_test, classical_probabilities)
    print(f"AUC: {auc_classical:.4f}")
    # Quantum
    print("\n2. Quantum Fraud Detection:")
    quantum_detector = QuantumFraudDetection(num_qubits=4)
    quantum_predictions, quantum_probabilities = quantum_detector.quantum_fraud_prediction(X_train_scaled, y_train, X_test_scaled)
    print(classification_report(y_test[:len(quantum_predictions)], quantum_predictions))
    auc_quantum = roc_auc_score(y_test[:len(quantum_predictions)], quantum_probabilities)
    print(f"AUC: {auc_quantum:.4f}")
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(classical_probabilities, bins=20, alpha=0.7, label='Classical', density=True)
    plt.hist(quantum_probabilities, bins=20, alpha=0.7, label='Quantum', density=True)
    plt.xlabel('Fraud Probability')
    plt.ylabel('Density')
    plt.title('Fraud Probability Distribution')
    plt.legend()
    plt.subplot(1, 2, 2)
    methods = ['Classical', 'Quantum']
    aucs = [auc_classical, auc_quantum]
    plt.bar(methods, aucs, alpha=0.7)
    plt.ylabel('AUC')
    plt.title('AUC Comparison')
    plt.tight_layout()
    plt.show()
    return {'auc_classical': auc_classical, 'auc_quantum': auc_quantum}

# Run demo
if __name__ == "__main__":
    fraud_results = compare_fraud_detection()
```

## 📊 Kết quả và Phân tích

### **Quantum Credit Fraud Detection Advantages:**

#### **1. Quantum Properties:**
- **Superposition**: Parallel fraud evaluation
- **Entanglement**: Complex feature correlations
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Fraud-specific Benefits:**
- **Non-linear Detection**: Quantum circuits capture complex fraud patterns
- **High-dimensional Features**: Handle many features efficiently
- **Quantum Advantage**: Potential speedup for large datasets

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Fraud Calibration**
Implement quantum fraud calibration methods.

### **Exercise 2: Quantum Fraud Ensemble**
Build quantum fraud ensemble methods.

### **Exercise 3: Quantum Fraud Validation**
Develop quantum fraud validation framework.

### **Exercise 4: Quantum Fraud Optimization**
Create quantum fraud optimization.

---

> *"Quantum credit fraud detection leverages quantum superposition and entanglement to provide superior fraud detection capabilities."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Credit Explainability](Day29.md) 