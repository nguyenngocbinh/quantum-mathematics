# Ngày 24: Quantum Credit Rating Models

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum credit rating models và classical rating models
- Nắm vững cách quantum computing cải thiện credit rating
- Implement quantum credit rating algorithms
- So sánh performance giữa quantum và classical rating models

## 📚 Lý thuyết

### **Credit Rating Fundamentals**

#### **1. Classical Credit Rating**

**Rating Models:**
- **Logistic Regression**: Linear probability models
- **Decision Trees**: Tree-based classification
- **Neural Networks**: Deep learning approaches
- **Support Vector Machines**: Margin-based classification

**Rating Metrics:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

#### **2. Quantum Credit Rating**

**Quantum State Encoding:**
```
|ψ⟩ = Σᵢ αᵢ|featureᵢ⟩
```

**Quantum Rating Operator:**
```
H_rating = Σᵢ Weightᵢ × Featureᵢ × |ratingᵢ⟩⟨ratingᵢ|
```

**Quantum Classification:**
```
Rating_quantum = argmax(⟨ψ|H_rating|ψ⟩)
```

### **Quantum Rating Methods**

#### **1. Quantum Feature Encoding:**
- **Quantum Feature Maps**: Encode features as quantum states
- **Quantum Amplitude Encoding**: Encode data in amplitudes
- **Quantum Kernel Methods**: Quantum kernel functions

#### **2. Quantum Classification:**
- **Quantum Support Vector Machines**: Quantum SVM implementation
- **Quantum Neural Networks**: Quantum neural network models
- **Quantum Decision Trees**: Quantum tree-based models

#### **3. Quantum Rating Calibration:**
- **Quantum Probability Estimation**: Quantum probability calculation
- **Quantum Confidence Intervals**: Quantum uncertainty quantification
- **Quantum Rating Stability**: Quantum rating stability analysis

## 💻 Thực hành

### **Project 24: Quantum Credit Rating Framework**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit_machine_learning.kernels import QuantumKernel

class ClassicalCreditRating:
    """Classical credit rating models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def generate_credit_data(self, n_samples=1000):
        """Generate synthetic credit data"""
        np.random.seed(42)
        
        # Generate features
        income = np.random.normal(50000, 20000, n_samples)
        debt = np.random.normal(30000, 15000, n_samples)
        credit_score = np.random.normal(700, 100, n_samples)
        payment_history = np.random.normal(650, 50, n_samples)
        employment_years = np.random.exponential(5, n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'income': income,
            'debt': debt,
            'credit_score': credit_score,
            'payment_history': payment_history,
            'employment_years': employment_years
        })
        
        # Generate default labels (simplified)
        default_prob = 1 / (1 + np.exp(-(-2 + 0.00001*income - 0.00002*debt + 0.01*credit_score)))
        default = np.random.binomial(1, default_prob)
        
        return data, default
    
    def logistic_regression_rating(self, X_train, y_train, X_test):
        """Logistic regression credit rating"""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        return predictions, probabilities

class QuantumCreditRating:
    """Quantum credit rating implementation"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.optimizer = SPSA(maxiter=100)
        
    def create_rating_circuit(self, features):
        """Create quantum circuit for credit rating"""
        feature_map = ZZFeatureMap(feature_dimension=len(features), reps=2)
        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=3)
        circuit = feature_map.compose(ansatz)
        return circuit
    
    def quantum_rating_prediction(self, X_train, y_train, X_test):
        """Quantum credit rating prediction"""
        predictions = []
        probabilities = []
        
        for i, sample in enumerate(X_test):
            if i >= 50:  # Limit for demonstration
                break
                
            # Create quantum circuit
            circuit = self.create_rating_circuit(sample)
            
            # Execute circuit
            job = execute(circuit, self.backend, shots=1000)
            result = job.result()
            counts = result.get_counts()
            
            # Extract prediction
            prediction, probability = self._extract_prediction_from_counts(counts)
            predictions.append(prediction)
            probabilities.append(probability)
        
        return np.array(predictions), np.array(probabilities)
    
    def _extract_prediction_from_counts(self, counts):
        """Extract prediction from quantum measurement counts"""
        total_shots = sum(counts.values())
        
        # Calculate default probability
        default_prob = 0.0
        for bitstring, count in counts.items():
            probability = count / total_shots
            # Use parity as default indicator
            parity = sum(int(bit) for bit in bitstring) % 2
            if parity == 1:
                default_prob += probability
        
        # Convert to prediction
        prediction = 1 if default_prob > 0.5 else 0
        
        return prediction, default_prob

def compare_credit_rating():
    """Compare classical and quantum credit rating"""
    print("=== Classical vs Quantum Credit Rating ===\n")
    
    # Generate data
    classical_rater = ClassicalCreditRating()
    data, labels = classical_rater.generate_credit_data(n_samples=500)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.3, random_state=42
    )
    
    # Scale features
    X_train_scaled = classical_rater.scaler.fit_transform(X_train)
    X_test_scaled = classical_rater.scaler.transform(X_test)
    
    # Classical rating
    print("1. Classical Credit Rating:")
    classical_predictions, classical_probabilities = classical_rater.logistic_regression_rating(
        X_train_scaled, y_train, X_test_scaled
    )
    
    print("Classification Report:")
    print(classification_report(y_test, classical_predictions))
    
    # Quantum rating
    print("\n2. Quantum Credit Rating:")
    quantum_rater = QuantumCreditRating(num_qubits=4)
    quantum_predictions, quantum_probabilities = quantum_rater.quantum_rating_prediction(
        X_train_scaled, y_train, X_test_scaled
    )
    
    # Compare results
    print("Quantum Classification Report:")
    print(classification_report(y_test[:len(quantum_predictions)], quantum_predictions))
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Confusion matrices
    plt.subplot(2, 3, 1)
    cm_classical = confusion_matrix(y_test, classical_predictions)
    sns.heatmap(cm_classical, annot=True, fmt='d', cmap='Blues')
    plt.title('Classical Confusion Matrix')
    
    plt.subplot(2, 3, 2)
    cm_quantum = confusion_matrix(y_test[:len(quantum_predictions)], quantum_predictions)
    sns.heatmap(cm_quantum, annot=True, fmt='d', cmap='Oranges')
    plt.title('Quantum Confusion Matrix')
    
    # Probability comparison
    plt.subplot(2, 3, 3)
    plt.scatter(classical_probabilities[:len(quantum_probabilities)], 
                quantum_probabilities, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Classical Probability')
    plt.ylabel('Quantum Probability')
    plt.title('Probability Comparison')
    plt.grid(True)
    
    # Rating accuracy over time
    plt.subplot(2, 3, 4)
    classical_accuracy = np.mean(classical_predictions == y_test)
    quantum_accuracy = np.mean(quantum_predictions == y_test[:len(quantum_predictions)])
    
    methods = ['Classical', 'Quantum']
    accuracies = [classical_accuracy, quantum_accuracy]
    
    plt.bar(methods, accuracies, color=['blue', 'orange'], alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Rating Accuracy Comparison')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'classical_predictions': classical_predictions,
        'quantum_predictions': quantum_predictions,
        'classical_probabilities': classical_probabilities,
        'quantum_probabilities': quantum_probabilities
    }

# Run demo
if __name__ == "__main__":
    rating_results = compare_credit_rating()
```

## 📊 Kết quả và Phân tích

### **Quantum Credit Rating Advantages:**

#### **1. Quantum Properties:**
- **Superposition**: Parallel feature evaluation
- **Entanglement**: Complex feature correlations
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Rating-specific Benefits:**
- **Non-linear Classification**: Quantum circuits capture complex rating relationships
- **High-dimensional Features**: Handle many features efficiently
- **Quantum Advantage**: Potential speedup for large datasets

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Rating Calibration**
Implement quantum rating calibration methods.

### **Exercise 2: Quantum Rating Ensemble**
Build quantum rating ensemble methods.

### **Exercise 3: Quantum Rating Validation**
Develop quantum rating validation framework.

### **Exercise 4: Quantum Rating Stability**
Create quantum rating stability analysis.

---

> *"Quantum credit rating leverages quantum superposition and entanglement to provide superior classification accuracy."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Credit Scoring](Day25.md) 