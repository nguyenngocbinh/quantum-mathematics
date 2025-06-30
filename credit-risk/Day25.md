# Ngày 25: Quantum Credit Scoring

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum credit scoring và classical credit scoring
- Nắm vững cách quantum computing cải thiện credit scoring
- Implement quantum credit scoring algorithms
- So sánh performance giữa quantum và classical scoring methods

## 📚 Lý thuyết

### **Credit Scoring Fundamentals**

#### **1. Classical Credit Scoring**

**Scoring Models:**
- **FICO Score**: Traditional credit scoring
- **Logistic Regression**: Linear scoring models
- **Random Forest**: Ensemble scoring methods
- **Gradient Boosting**: Advanced scoring algorithms

**Scoring Metrics:**
```
Score = Σᵢ Weightᵢ × Featureᵢ
AUC = Area under ROC curve
KS = Kolmogorov-Smirnov statistic
```

#### **2. Quantum Credit Scoring**

**Quantum Score Encoding:**
```
|ψ⟩ = Σᵢ αᵢ|featureᵢ⟩
```

**Quantum Scoring Operator:**
```
H_score = Σᵢ Weightᵢ × Featureᵢ × |scoreᵢ⟩⟨scoreᵢ|
```

**Quantum Score Calculation:**
```
Score_quantum = ⟨ψ|H_score|ψ⟩
```

### **Quantum Scoring Methods**

#### **1. Quantum Feature Engineering:**
- **Quantum Feature Selection**: Quantum feature selection methods
- **Quantum Feature Transformation**: Quantum feature transformation
- **Quantum Feature Interaction**: Quantum feature interaction modeling

#### **2. Quantum Score Calculation:**
- **Quantum Linear Scoring**: Quantum linear scoring models
- **Quantum Non-linear Scoring**: Quantum non-linear scoring
- **Quantum Ensemble Scoring**: Quantum ensemble methods

#### **3. Quantum Score Calibration:**
- **Quantum Score Calibration**: Quantum score calibration methods
- **Quantum Score Validation**: Quantum score validation
- **Quantum Score Stability**: Quantum score stability analysis

## 💻 Thực hành

### **Project 25: Quantum Credit Scoring Framework**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA, COBYLA

class ClassicalCreditScoring:
    """Classical credit scoring models"""
    
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
        
        # Generate default labels
        default_prob = 1 / (1 + np.exp(-(-2 + 0.00001*income - 0.00002*debt + 0.01*credit_score)))
        default = np.random.binomial(1, default_prob)
        
        return data, default
    
    def logistic_regression_scoring(self, X_train, y_train, X_test):
        """Logistic regression credit scoring"""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        
        return scores

class QuantumCreditScoring:
    """Quantum credit scoring implementation"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.optimizer = SPSA(maxiter=100)
        
    def create_scoring_circuit(self, features):
        """Create quantum circuit for credit scoring"""
        feature_map = ZZFeatureMap(feature_dimension=len(features), reps=2)
        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=3)
        circuit = feature_map.compose(ansatz)
        return circuit
    
    def quantum_scoring(self, X_train, y_train, X_test):
        """Quantum credit scoring"""
        scores = []
        
        for i, sample in enumerate(X_test):
            if i >= 50:  # Limit for demonstration
                break
                
            # Create quantum circuit
            circuit = self.create_scoring_circuit(sample)
            
            # Execute circuit
            job = execute(circuit, self.backend, shots=1000)
            result = job.result()
            counts = result.get_counts()
            
            # Extract score
            score = self._extract_score_from_counts(counts)
            scores.append(score)
        
        return np.array(scores)
    
    def _extract_score_from_counts(self, counts):
        """Extract score from quantum measurement counts"""
        total_shots = sum(counts.values())
        
        # Calculate quantum score
        score = 0.0
        for bitstring, count in counts.items():
            probability = count / total_shots
            # Use parity as score indicator
            parity = sum(int(bit) for bit in bitstring) % 2
            score += probability * (1 if parity == 0 else 0)
        
        return score

def compare_credit_scoring():
    """Compare classical and quantum credit scoring"""
    print("=== Classical vs Quantum Credit Scoring ===\n")
    
    # Generate data
    classical_scorer = ClassicalCreditScoring()
    data, labels = classical_scorer.generate_credit_data(n_samples=500)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.3, random_state=42
    )
    
    # Scale features
    X_train_scaled = classical_scorer.scaler.fit_transform(X_train)
    X_test_scaled = classical_scorer.scaler.transform(X_test)
    
    # Classical scoring
    print("1. Classical Credit Scoring:")
    classical_scores = classical_scorer.logistic_regression_scoring(
        X_train_scaled, y_train, X_test_scaled
    )
    
    classical_auc = roc_auc_score(y_test, classical_scores)
    print(f"   Classical AUC: {classical_auc:.4f}")
    
    # Quantum scoring
    print("\n2. Quantum Credit Scoring:")
    quantum_scorer = QuantumCreditScoring(num_qubits=4)
    quantum_scores = quantum_scorer.quantum_scoring(
        X_train_scaled, y_train, X_test_scaled
    )
    
    quantum_auc = roc_auc_score(y_test[:len(quantum_scores)], quantum_scores)
    print(f"   Quantum AUC: {quantum_auc:.4f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # ROC curves
    plt.subplot(2, 3, 1)
    fpr_classical, tpr_classical, _ = roc_curve(y_test, classical_scores)
    fpr_quantum, tpr_quantum, _ = roc_curve(y_test[:len(quantum_scores)], quantum_scores)
    
    plt.plot(fpr_classical, tpr_classical, label=f'Classical (AUC={classical_auc:.3f})')
    plt.plot(fpr_quantum, tpr_quantum, label=f'Quantum (AUC={quantum_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    
    # Score distributions
    plt.subplot(2, 3, 2)
    plt.hist(classical_scores, bins=20, alpha=0.7, label='Classical', density=True)
    plt.hist(quantum_scores, bins=20, alpha=0.7, label='Quantum', density=True)
    plt.xlabel('Credit Score')
    plt.ylabel('Density')
    plt.title('Score Distributions')
    plt.legend()
    plt.grid(True)
    
    # Score comparison
    plt.subplot(2, 3, 3)
    plt.scatter(classical_scores[:len(quantum_scores)], quantum_scores, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Classical Score')
    plt.ylabel('Quantum Score')
    plt.title('Score Comparison')
    plt.grid(True)
    
    # AUC comparison
    plt.subplot(2, 3, 4)
    methods = ['Classical', 'Quantum']
    aucs = [classical_auc, quantum_auc]
    
    plt.bar(methods, aucs, color=['blue', 'orange'], alpha=0.7)
    plt.ylabel('AUC')
    plt.title('AUC Comparison')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'classical_scores': classical_scores,
        'quantum_scores': quantum_scores,
        'classical_auc': classical_auc,
        'quantum_auc': quantum_auc
    }

# Run demo
if __name__ == "__main__":
    scoring_results = compare_credit_scoring()
```

## 📊 Kết quả và Phân tích

### **Quantum Credit Scoring Advantages:**

#### **1. Quantum Properties:**
- **Superposition**: Parallel feature evaluation
- **Entanglement**: Complex feature correlations
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Scoring-specific Benefits:**
- **Non-linear Scoring**: Quantum circuits capture complex scoring relationships
- **High-dimensional Features**: Handle many features efficiently
- **Quantum Advantage**: Potential speedup for large datasets

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Score Calibration**
Implement quantum score calibration methods.

### **Exercise 2: Quantum Score Ensemble**
Build quantum score ensemble methods.

### **Exercise 3: Quantum Score Validation**
Develop quantum score validation framework.

### **Exercise 4: Quantum Score Stability**
Create quantum score stability analysis.

---

> *"Quantum credit scoring leverages quantum superposition and entanglement to provide superior scoring accuracy."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Credit Monitoring](Day26.md) 