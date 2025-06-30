# Ngày 6: Quantum Probability và Finance

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum probability và cách nó khác biệt với classical probability
- Nắm vững cách quantum probability áp dụng cho financial modeling
- Implement quantum probability models cho credit risk assessment
- So sánh quantum và classical probability approaches

## 📚 Lý thuyết

### **Quantum Probability Fundamentals**

#### **1. Classical vs Quantum Probability**

**Classical Probability:**
- Based on Kolmogorov axioms
- Additive: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
- Commutative: P(A ∩ B) = P(B ∩ A)
- Real-valued probabilities

**Quantum Probability:**
- Based on quantum mechanics principles
- Non-additive trong certain contexts
- Non-commutative observables
- Complex-valued amplitudes

#### **2. Quantum Probability Axioms**

**Born Rule:**
```
P(ψ → φ) = |⟨φ|ψ⟩|²
```

**Quantum State:**
```
|ψ⟩ = Σᵢ cᵢ|i⟩
```

**Probability Amplitude:**
```
cᵢ = ⟨i|ψ⟩
```

#### **3. Quantum Interference**

**Superposition Principle:**
```
|ψ⟩ = α|0⟩ + β|1⟩
P(0) = |α|², P(1) = |β|²
```

**Interference Effects:**
```
P(ψ → φ) = |Σᵢ cᵢ⟨φ|i⟩|² ≠ Σᵢ |cᵢ⟨φ|i⟩|²
```

### **Quantum Probability cho Finance**

#### **1. Financial Applications:**

**Credit Risk Assessment:**
- Quantum superposition của default states
- Interference effects trong risk correlations
- Non-commutative risk measures

**Portfolio Optimization:**
- Quantum states cho asset allocations
- Entanglement cho correlation modeling
- Quantum uncertainty principles

**Market Modeling:**
- Quantum random walks
- Superposition của market scenarios
- Quantum measurement effects

#### **2. Quantum Financial States:**

**Credit State Representation:**
```
|credit⟩ = α|good⟩ + β|default⟩
```

**Portfolio State:**
```
|portfolio⟩ = Σᵢ wᵢ|asset_i⟩
```

**Market State:**
```
|market⟩ = Σₛ pₛ|scenario_s⟩
```

#### **3. Quantum Risk Measures:**

**Quantum VaR:**
```
QVaR = min{V : P(portfolio_loss > V) ≤ α}
```

**Quantum CVaR:**
```
QCVaR = E[loss | loss > QVaR]
```

**Quantum Entropy:**
```
S(ρ) = -Tr(ρ log ρ)
```

### **Quantum Probability Advantages**

#### **1. Non-linearity:**
- Captures complex market dynamics
- Models non-linear correlations
- Handles regime changes

#### **2. Interference Effects:**
- Market sentiment interactions
- Cross-asset correlations
- Systemic risk modeling

#### **3. Uncertainty Principles:**
- Risk-return trade-offs
- Measurement effects
- Information limits

## 💻 Thực hành

### **Project 6: Quantum Probability Framework cho Finance**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import ZZFeatureMap
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA
import pennylane as qml

class QuantumProbabilityFramework:
    """Quantum probability framework for financial applications"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('statevector_simulator')
        
    def create_quantum_state(self, probabilities):
        """
        Create quantum state from classical probabilities
        """
        # Normalize probabilities
        probs = np.array(probabilities)
        probs = probs / np.sum(probs)
        
        # Create quantum circuit
        circuit = QuantumCircuit(self.num_qubits)
        
        # Encode probabilities into quantum state
        for i, prob in enumerate(probs[:2**self.num_qubits]):
            if prob > 0:
                # Convert probability to amplitude
                amplitude = np.sqrt(prob)
                # Apply rotation to encode amplitude
                circuit.rx(2 * np.arccos(amplitude), i % self.num_qubits)
        
        return circuit
    
    def measure_quantum_probability(self, circuit, shots=1000):
        """
        Measure quantum probability distribution
        """
        # Add measurement
        circuit.measure_all()
        
        # Execute circuit
        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Convert counts to probabilities
        total_shots = sum(counts.values())
        probabilities = {}
        
        for state, count in counts.items():
            probabilities[state] = count / total_shots
            
        return probabilities
    
    def quantum_interference_demo(self):
        """
        Demonstrate quantum interference effects
        """
        # Create two quantum states
        circuit1 = QuantumCircuit(2, 2)
        circuit1.h(0)  # Hadamard gate creates superposition
        circuit1.cx(0, 1)  # Entanglement
        
        circuit2 = QuantumCircuit(2, 2)
        circuit2.x(0)  # Flip first qubit
        circuit2.h(0)  # Hadamard gate
        circuit2.cx(0, 1)  # Entanglement
        
        # Measure both circuits
        prob1 = self.measure_quantum_probability(circuit1)
        prob2 = self.measure_quantum_probability(circuit2)
        
        print("Quantum State 1 Probabilities:")
        for state, prob in prob1.items():
            print(f"  |{state}⟩: {prob:.4f}")
            
        print("\nQuantum State 2 Probabilities:")
        for state, prob in prob2.items():
            print(f"  |{state}⟩: {prob:.4f}")
        
        return prob1, prob2

class QuantumCreditProbability:
    """Quantum probability model for credit risk"""
    
    def __init__(self):
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_credit_state(self, credit_score, risk_factors):
        """
        Create quantum state for credit assessment
        """
        num_qubits = len(risk_factors) + 1  # +1 for credit score
        circuit = QuantumCircuit(num_qubits, num_qubits)
        
        # Encode credit score
        normalized_score = min(credit_score / 850, 1.0)
        circuit.rx(normalized_score * np.pi, 0)
        
        # Encode risk factors
        for i, factor in enumerate(risk_factors):
            normalized_factor = min(factor, 1.0)
            circuit.rx(normalized_factor * np.pi, i + 1)
        
        # Add entanglement between credit score and risk factors
        for i in range(1, num_qubits):
            circuit.cx(0, i)
        
        return circuit
    
    def calculate_default_probability(self, credit_score, risk_factors, shots=1000):
        """
        Calculate quantum default probability
        """
        circuit = self.create_credit_state(credit_score, risk_factors)
        
        # Add measurement
        circuit.measure_all()
        
        # Execute circuit
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate default probability
        default_prob = self._extract_default_probability(counts)
        return default_prob
    
    def _extract_default_probability(self, counts):
        """
        Extract default probability from measurement counts
        """
        total_shots = sum(counts.values())
        default_count = 0
        
        for state, count in counts.items():
            # Consider states with more 1s as higher default risk
            ones_count = state.count('1')
            if ones_count > len(state) / 2:
                default_count += count
        
        return default_count / total_shots

class QuantumPortfolioProbability:
    """Quantum probability model for portfolio optimization"""
    
    def __init__(self, num_assets=4):
        self.num_assets = num_assets
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_portfolio_state(self, weights, returns, volatilities):
        """
        Create quantum state for portfolio
        """
        num_qubits = self.num_assets * 2  # 2 qubits per asset for return and volatility
        circuit = QuantumCircuit(num_qubits, num_qubits)
        
        # Encode portfolio parameters
        for i in range(self.num_assets):
            # Encode weight
            weight_qubit = i * 2
            normalized_weight = min(weights[i], 1.0)
            circuit.rx(normalized_weight * np.pi, weight_qubit)
            
            # Encode return
            return_qubit = i * 2 + 1
            normalized_return = (returns[i] - min(returns)) / (max(returns) - min(returns))
            circuit.ry(normalized_return * np.pi, return_qubit)
            
            # Add entanglement between weight and return
            circuit.cx(weight_qubit, return_qubit)
        
        # Add entanglement between assets
        for i in range(self.num_assets - 1):
            circuit.cx(i * 2, (i + 1) * 2)
        
        return circuit
    
    def calculate_portfolio_risk(self, weights, returns, volatilities, shots=1000):
        """
        Calculate quantum portfolio risk
        """
        circuit = self.create_portfolio_state(weights, returns, volatilities)
        
        # Add measurement
        circuit.measure_all()
        
        # Execute circuit
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate portfolio risk
        risk_measure = self._extract_portfolio_risk(counts, weights, returns, volatilities)
        return risk_measure
    
    def _extract_portfolio_risk(self, counts, weights, returns, volatilities):
        """
        Extract portfolio risk from measurement counts
        """
        total_shots = sum(counts.values())
        risk_sum = 0
        
        for state, count in counts.items():
            # Calculate risk contribution from this state
            state_risk = self._calculate_state_risk(state, weights, returns, volatilities)
            risk_sum += state_risk * count
        
        return risk_sum / total_shots
    
    def _calculate_state_risk(self, state, weights, returns, volatilities):
        """
        Calculate risk contribution from a quantum state
        """
        # Parse state into asset contributions
        risk_contributions = []
        
        for i in range(self.num_assets):
            weight_bit = int(state[i * 2])
            return_bit = int(state[i * 2 + 1])
            
            # Calculate contribution based on bits
            weight_factor = weights[i] * (1 + 0.1 * weight_bit)
            return_factor = returns[i] * (1 + 0.1 * return_bit)
            vol_factor = volatilities[i]
            
            contribution = weight_factor * return_factor * vol_factor
            risk_contributions.append(contribution)
        
        # Total portfolio risk
        total_risk = np.sqrt(np.sum(np.array(risk_contributions) ** 2))
        return total_risk

def quantum_probability_demo():
    """
    Demo quantum probability concepts
    """
    print("=== Quantum Probability Demo ===\n")
    
    # Initialize framework
    qpf = QuantumProbabilityFramework(num_qubits=2)
    
    # Demo 1: Quantum interference
    print("1. Quantum Interference Effects:")
    prob1, prob2 = qpf.quantum_interference_demo()
    
    # Demo 2: Credit probability
    print("\n2. Quantum Credit Probability:")
    qcp = QuantumCreditProbability()
    
    credit_score = 750
    risk_factors = [0.3, 0.7, 0.5, 0.2]
    
    default_prob = qcp.calculate_default_probability(credit_score, risk_factors)
    print(f"Credit Score: {credit_score}")
    print(f"Risk Factors: {risk_factors}")
    print(f"Quantum Default Probability: {default_prob:.4f}")
    
    # Demo 3: Portfolio probability
    print("\n3. Quantum Portfolio Probability:")
    qpp = QuantumPortfolioProbability(num_assets=4)
    
    weights = [0.25, 0.25, 0.25, 0.25]
    returns = [0.08, 0.12, 0.06, 0.10]
    volatilities = [0.15, 0.20, 0.12, 0.18]
    
    portfolio_risk = qpp.calculate_portfolio_risk(weights, returns, volatilities)
    print(f"Weights: {weights}")
    print(f"Returns: {returns}")
    print(f"Volatilities: {volatilities}")
    print(f"Quantum Portfolio Risk: {portfolio_risk:.4f}")
    
    return prob1, prob2, default_prob, portfolio_risk

# Exercise: Quantum Probability vs Classical Probability
def quantum_vs_classical_probability():
    """
    Exercise: Compare quantum and classical probability approaches
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate credit events
    credit_scores = np.random.normal(700, 100, n_samples)
    default_events = np.random.binomial(1, 0.05, n_samples)  # 5% default rate
    
    # Classical probability calculation
    classical_probs = []
    for score in credit_scores:
        # Simple logistic model
        prob = 1 / (1 + np.exp(-(score - 700) / 100))
        classical_probs.append(prob)
    
    # Quantum probability calculation
    quantum_probs = []
    qcp = QuantumCreditProbability()
    
    for score in credit_scores[:100]:  # Limit for computational efficiency
        risk_factors = np.random.uniform(0, 1, 4)
        q_prob = qcp.calculate_default_probability(score, risk_factors)
        quantum_probs.append(q_prob)
    
    # Compare results
    classical_mean = np.mean(classical_probs)
    quantum_mean = np.mean(quantum_probs)
    
    print("=== Quantum vs Classical Probability Comparison ===")
    print(f"Classical Mean Probability: {classical_mean:.4f}")
    print(f"Quantum Mean Probability: {quantum_mean:.4f}")
    print(f"Difference: {abs(classical_mean - quantum_mean):.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(classical_probs, bins=30, alpha=0.7, label='Classical')
    plt.xlabel('Default Probability')
    plt.ylabel('Frequency')
    plt.title('Classical Probability Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(quantum_probs, bins=30, alpha=0.7, label='Quantum', color='orange')
    plt.xlabel('Default Probability')
    plt.ylabel('Frequency')
    plt.title('Quantum Probability Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return classical_probs, quantum_probs

# Run demos
if __name__ == "__main__":
    print("Running Quantum Probability Demos...")
    prob1, prob2, default_prob, portfolio_risk = quantum_probability_demo()
    
    print("\nRunning Quantum vs Classical Comparison...")
    classical_probs, quantum_probs = quantum_vs_classical_probability()
```

### **Exercise 1: Quantum Entropy và Information**

```python
def quantum_entropy_exercise():
    """
    Exercise: Calculate quantum entropy for financial states
    """
    from qiskit.quantum_info import entropy
    
    # Create different quantum states
    states = []
    
    # Pure state
    pure_state = Statevector([1, 0, 0, 0])
    states.append(("Pure State", pure_state))
    
    # Mixed state
    mixed_state = Statevector([0.5, 0.5, 0, 0])
    states.append(("Mixed State", mixed_state))
    
    # Maximally mixed state
    max_mixed = Statevector([0.5, 0.5, 0.5, 0.5])
    states.append(("Maximally Mixed", max_mixed))
    
    # Calculate entropy for each state
    print("=== Quantum Entropy Analysis ===")
    for name, state in states:
        # Convert to density matrix
        rho = state.to_operator()
        ent = entropy(rho)
        print(f"{name}: {ent:.4f}")
    
    return states

def quantum_information_flow():
    """
    Exercise: Analyze quantum information flow in financial systems
    """
    # Create quantum circuit for information flow
    circuit = QuantumCircuit(4, 4)
    
    # Initial state encoding financial information
    circuit.h(0)  # Market sentiment
    circuit.h(1)  # Credit conditions
    circuit.cx(0, 2)  # Market affects credit
    circuit.cx(1, 3)  # Credit affects risk
    
    # Add measurement
    circuit.measure_all()
    
    # Execute circuit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    # Analyze information flow
    print("=== Quantum Information Flow Analysis ===")
    print("Measurement Results:")
    for state, count in counts.items():
        print(f"  |{state}⟩: {count}")
    
    # Calculate mutual information
    mutual_info = calculate_mutual_information(counts)
    print(f"\nMutual Information: {mutual_info:.4f}")
    
    return circuit, counts

def calculate_mutual_information(counts):
    """
    Calculate mutual information from measurement counts
    """
    total_shots = sum(counts.values())
    
    # Calculate marginal probabilities
    p_market = {}
    p_credit = {}
    
    for state, count in counts.items():
        market_bit = state[0]
        credit_bit = state[1]
        
        p_market[market_bit] = p_market.get(market_bit, 0) + count
        p_credit[credit_bit] = p_credit.get(credit_bit, 0) + count
    
    # Normalize
    for bit in p_market:
        p_market[bit] /= total_shots
    for bit in p_credit:
        p_credit[bit] /= total_shots
    
    # Calculate mutual information
    mi = 0
    for state, count in counts.items():
        p_joint = count / total_shots
        p_m = p_market[state[0]]
        p_c = p_credit[state[1]]
        
        if p_joint > 0 and p_m > 0 and p_c > 0:
            mi += p_joint * np.log2(p_joint / (p_m * p_c))
    
    return mi

# Run exercises
if __name__ == "__main__":
    print("Running Quantum Entropy Exercise...")
    states = quantum_entropy_exercise()
    
    print("\nRunning Quantum Information Flow Exercise...")
    circuit, counts = quantum_information_flow()
```

## 📊 Kết quả và Phân tích

### **Quantum Probability Advantages:**

#### **1. Non-linearity:**
- **Interference Effects**: Captures complex market interactions
- **Superposition**: Parallel processing of multiple scenarios
- **Entanglement**: Models correlated financial events

#### **2. Information Processing:**
- **Quantum Entropy**: Better measure of uncertainty
- **Mutual Information**: Captures non-linear correlations
- **Quantum Channels**: Models information flow in markets

#### **3. Financial Applications:**
- **Credit Risk**: Quantum superposition of default states
- **Portfolio Risk**: Entangled asset correlations
- **Market Modeling**: Quantum random walks

### **Comparison với Classical Probability:**

#### **Classical Limitations:**
- Linear correlations
- Gaussian assumptions
- Additive probability measures

#### **Quantum Advantages:**
- Non-linear correlations
- Non-Gaussian distributions
- Interference effects

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Probability Calibration**
Implement quantum probability calibration methods cho financial models.

### **Exercise 2: Quantum Entropy Optimization**
Build quantum entropy optimization cho portfolio management.

### **Exercise 3: Quantum Information Theory cho Finance**
Develop quantum information theory applications cho credit risk.

### **Exercise 4: Quantum Probability Validation**
Create validation framework cho quantum probability models.

---

> *"Quantum probability provides a more nuanced view of financial uncertainty, capturing the complex, non-linear nature of market dynamics."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Random Walks cho Market Modeling](Day7.md) 