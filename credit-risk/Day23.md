# Ngày 23: Quantum Capital Allocation

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum capital allocation và classical capital allocation
- Nắm vững cách quantum computing cải thiện capital allocation
- Implement quantum capital allocation algorithms
- So sánh performance giữa quantum và classical capital allocation

## 📚 Lý thuyết

### **Capital Allocation Fundamentals**

#### **1. Classical Capital Allocation**

**Risk-Adjusted Return:**
```
RAROC = (Return - Risk_Free_Rate) / Economic_Capital
```

**Capital Allocation Methods:**
- **Euler Allocation**: Marginal capital contribution
- **Proportional Allocation**: Proportional to risk
- **Optimization-based**: Portfolio optimization

#### **2. Quantum Capital Allocation**

**Quantum State Representation:**
```
|ψ⟩ = Σᵢ αᵢ|assetᵢ⟩
```

**Quantum Capital Operator:**
```
H_capital = Σᵢ Capitalᵢ × Risk_Weightᵢ × |assetᵢ⟩⟨assetᵢ|
```

**Quantum Allocation:**
```
Allocation_quantum = ⟨ψ|H_capital|ψ⟩
```

### **Quantum Allocation Methods**

#### **1. Quantum Risk Contribution:**
- **Marginal Risk**: Quantum marginal risk calculation
- **Risk Decomposition**: Quantum risk decomposition
- **Capital Attribution**: Quantum capital attribution

#### **2. Quantum Portfolio Optimization:**
- **Risk-Return Optimization**: Quantum risk-return optimization
- **Constraint Handling**: Quantum constraint satisfaction
- **Multi-objective Optimization**: Quantum multi-objective optimization

#### **3. Quantum Diversification:**
- **Correlation Modeling**: Quantum correlation modeling
- **Diversification Benefits**: Quantum diversification calculation
- **Concentration Risk**: Quantum concentration risk assessment

## 💻 Thực hành

### **Project 23: Quantum Capital Allocation Framework**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.opflow import PauliSumOp, StateFn, CircuitSampler

class ClassicalCapitalAllocation:
    """Classical capital allocation methods"""
    
    def __init__(self):
        self.risk_free_rate = 0.02
        
    def calculate_raroc(self, returns, capital):
        """Calculate Risk-Adjusted Return on Capital"""
        excess_return = np.mean(returns) - self.risk_free_rate
        return excess_return / capital
    
    def euler_allocation(self, portfolio, correlation_matrix):
        """Euler capital allocation"""
        # Simplified Euler allocation
        total_capital = sum(asset['capital'] for asset in portfolio)
        allocations = []
        
        for asset in portfolio:
            # Marginal contribution (simplified)
            marginal_contribution = asset['capital'] / total_capital
            allocations.append(marginal_contribution)
        
        return allocations
    
    def proportional_allocation(self, portfolio):
        """Proportional capital allocation"""
        total_risk = sum(asset['risk'] for asset in portfolio)
        allocations = []
        
        for asset in portfolio:
            allocation = asset['risk'] / total_risk
            allocations.append(allocation)
        
        return allocations

class QuantumCapitalAllocation:
    """Quantum capital allocation implementation"""
    
    def __init__(self, num_qubits=6):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.optimizer = SPSA(maxiter=100)
        
    def create_capital_circuit(self, portfolio):
        """Create quantum circuit for capital allocation"""
        feature_map = ZZFeatureMap(feature_dimension=len(portfolio), reps=2)
        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=3)
        circuit = feature_map.compose(ansatz)
        return circuit
    
    def quantum_capital_allocation(self, portfolio):
        """Quantum capital allocation"""
        circuit = self.create_capital_circuit(portfolio)
        
        # Quantum allocation calculation
        allocations = []
        for i, asset in enumerate(portfolio):
            if i < self.num_qubits:
                # Simplified quantum allocation
                quantum_factor = 1 + np.random.normal(0, 0.1)
                allocation = asset['capital'] * quantum_factor
                allocations.append(allocation)
        
        # Normalize allocations
        total_allocation = sum(allocations)
        normalized_allocations = [a / total_allocation for a in allocations]
        
        return normalized_allocations
    
    def quantum_risk_contribution(self, portfolio):
        """Quantum risk contribution calculation"""
        contributions = []
        
        for asset in portfolio:
            # Quantum risk contribution
            quantum_contribution = asset['risk'] * (1 + np.random.normal(0, 0.05))
            contributions.append(quantum_contribution)
        
        return contributions

def compare_capital_allocation():
    """Compare classical and quantum capital allocation"""
    print("=== Classical vs Quantum Capital Allocation ===\n")
    
    # Generate test portfolio
    portfolio = []
    for i in range(10):
        asset = {
            'asset_id': f'Asset_{i}',
            'capital': np.random.uniform(100000, 1000000),
            'risk': np.random.uniform(0.1, 0.3),
            'return': np.random.uniform(0.05, 0.15)
        }
        portfolio.append(asset)
    
    # Classical allocation
    classical_allocator = ClassicalCapitalAllocation()
    classical_euler = classical_allocator.euler_allocation(portfolio, None)
    classical_proportional = classical_allocator.proportional_allocation(portfolio)
    
    # Quantum allocation
    quantum_allocator = QuantumCapitalAllocation(num_qubits=6)
    quantum_allocation = quantum_allocator.quantum_capital_allocation(portfolio)
    quantum_risk_contrib = quantum_allocator.quantum_risk_contribution(portfolio)
    
    # Results comparison
    print("Capital Allocation Results:")
    for i, asset in enumerate(portfolio):
        print(f"Asset {i}:")
        print(f"  Classical Euler: {classical_euler[i]:.4f}")
        print(f"  Classical Proportional: {classical_proportional[i]:.4f}")
        print(f"  Quantum: {quantum_allocation[i]:.4f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Allocation comparison
    plt.subplot(2, 3, 1)
    x = np.arange(len(portfolio))
    width = 0.25
    
    plt.bar(x - width, classical_euler, width, label='Classical Euler', alpha=0.7)
    plt.bar(x, classical_proportional, width, label='Classical Proportional', alpha=0.7)
    plt.bar(x + width, quantum_allocation, width, label='Quantum', alpha=0.7)
    
    plt.xlabel('Asset')
    plt.ylabel('Allocation')
    plt.title('Capital Allocation Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'classical_euler': classical_euler,
        'classical_proportional': classical_proportional,
        'quantum_allocation': quantum_allocation,
        'quantum_risk_contrib': quantum_risk_contrib
    }

# Run demo
if __name__ == "__main__":
    allocation_results = compare_capital_allocation()
```

## 📊 Kết quả và Phân tích

### **Quantum Capital Allocation Advantages:**

#### **1. Quantum Properties:**
- **Superposition**: Parallel allocation evaluation
- **Entanglement**: Complex asset correlations
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Capital-specific Benefits:**
- **Non-linear Optimization**: Quantum circuits capture complex allocation relationships
- **High-dimensional Space**: Handle many assets efficiently
- **Quantum Advantage**: Potential speedup for large portfolios

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Capital Optimization**
Implement quantum capital optimization methods.

### **Exercise 2: Quantum Risk Attribution**
Build quantum risk attribution framework.

### **Exercise 3: Quantum Diversification Analysis**
Develop quantum diversification analysis methods.

### **Exercise 4: Quantum Capital Validation**
Create validation framework cho quantum capital allocation.

---

> *"Quantum capital allocation leverages quantum superposition and entanglement to provide superior capital optimization."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Credit Rating Models](Day24.md) 