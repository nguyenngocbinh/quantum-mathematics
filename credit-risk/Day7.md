# Ngày 7: Quantum Random Walks cho Market Modeling

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum random walks và classical random walks
- Nắm vững cách quantum random walks mô hình hóa thị trường tài chính
- Implement quantum random walk algorithms cho market simulation
- So sánh performance giữa quantum và classical random walks

## 📚 Lý thuyết

### **Random Walks Fundamentals**

#### **1. Classical Random Walks**

**One-dimensional Random Walk:**
```
X(t+1) = X(t) + ε(t)
ε(t) ~ N(0, σ²)
```

**Properties:**
- Linear growth in variance: Var(X(t)) = σ²t
- Gaussian distribution
- Markov property

#### **2. Quantum Random Walks**

**Quantum State:**
```
|ψ(t)⟩ = Σₓ cₓ(t)|x⟩ ⊗ |s⟩
```

**Evolution Operator:**
```
U = S · C
S: Shift operator
C: Coin operator
```

**Quantum Advantage:**
- Quadratic speedup cho certain problems
- Non-Gaussian distributions
- Quantum interference effects

### **Market Modeling với Random Walks**

#### **1. Classical Market Models:**

**Geometric Brownian Motion:**
```
dS = μSdt + σSdW
```

**Mean Reversion:**
```
dS = κ(θ - S)dt + σdW
```

**Jump Diffusion:**
```
dS = μSdt + σSdW + S(e^J - 1)dN
```

#### **2. Quantum Market Models:**

**Quantum Price Evolution:**
```
|price(t)⟩ = U^t|price(0)⟩
```

**Quantum Volatility:**
```
σ_quantum = √(⟨ψ|H²|ψ⟩ - ⟨ψ|H|ψ⟩²)
```

**Quantum Correlation:**
```
ρ_quantum = ⟨ψ₁|ψ₂⟩/√(⟨ψ₁|ψ₁⟩⟨ψ₂|ψ₂⟩)
```

### **Quantum Random Walk Types**

#### **1. Discrete-time Quantum Walks:**

**Coin-based Walks:**
```
|ψ(t+1)⟩ = S · C|ψ(t)⟩
```

**Properties:**
- Quadratic speedup
- Ballistic spreading
- Quantum interference

#### **2. Continuous-time Quantum Walks:**

**Hamiltonian Evolution:**
```
|ψ(t)⟩ = e^(-iHt)|ψ(0)⟩
```

**Properties:**
- Exponential speedup cho certain problems
- Continuous evolution
- Natural quantum dynamics

#### **3. Quantum Walks on Graphs:**

**Adjacency Matrix:**
```
H = -γA
A: Adjacency matrix
γ: Coupling strength
```

## 💻 Thực hành

### **Project 7: Quantum Random Walk Market Simulator**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import QFT
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA
import pennylane as qml

class ClassicalRandomWalk:
    """Classical random walk implementation"""
    
    def __init__(self, steps=100, drift=0.0, volatility=0.1):
        self.steps = steps
        self.drift = drift
        self.volatility = volatility
        
    def simulate_path(self, initial_price=100.0):
        """
        Simulate classical random walk path
        """
        prices = [initial_price]
        
        for _ in range(self.steps):
            # Generate random step
            step = np.random.normal(self.drift, self.volatility)
            
            # Update price
            new_price = prices[-1] * (1 + step)
            prices.append(new_price)
        
        return np.array(prices)
    
    def simulate_multiple_paths(self, n_paths=1000, initial_price=100.0):
        """
        Simulate multiple classical random walk paths
        """
        paths = []
        
        for _ in range(n_paths):
            path = self.simulate_path(initial_price)
            paths.append(path)
        
        return np.array(paths)

class QuantumRandomWalk:
    """Quantum random walk implementation"""
    
    def __init__(self, num_qubits=8):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('statevector_simulator')
        
    def create_coin_operator(self):
        """
        Create Hadamard coin operator
        """
        coin = QuantumCircuit(1)
        coin.h(0)
        return coin.to_gate()
    
    def create_shift_operator(self):
        """
        Create shift operator for quantum walk
        """
        shift = QuantumCircuit(self.num_qubits + 1)  # +1 for coin qubit
        
        # Controlled shift based on coin state
        for i in range(self.num_qubits):
            shift.cx(self.num_qubits, i)
        
        return shift.to_gate()
    
    def quantum_walk_step(self, circuit, step):
        """
        Apply one step of quantum random walk
        """
        # Apply coin operator
        coin_gate = self.create_coin_operator()
        circuit.append(coin_gate, [self.num_qubits])
        
        # Apply shift operator
        shift_gate = self.create_shift_operator()
        circuit.append(shift_gate, list(range(self.num_qubits + 1)))
        
        return circuit
    
    def simulate_quantum_walk(self, steps=50, initial_position=0):
        """
        Simulate quantum random walk
        """
        # Create quantum circuit
        circuit = QuantumCircuit(self.num_qubits + 1, self.num_qubits)
        
        # Initialize position
        if initial_position < self.num_qubits:
            circuit.x(initial_position)
        
        # Initialize coin in superposition
        circuit.h(self.num_qubits)
        
        # Apply quantum walk steps
        for step in range(steps):
            circuit = self.quantum_walk_step(circuit, step)
        
        # Measure position qubits
        circuit.measure(list(range(self.num_qubits)), list(range(self.num_qubits)))
        
        return circuit
    
    def get_position_distribution(self, circuit, shots=1000):
        """
        Get position distribution from quantum walk
        """
        # Execute circuit
        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Convert to position distribution
        positions = np.zeros(2**self.num_qubits)
        
        for state, count in counts.items():
            # Convert binary state to position
            pos = int(state, 2)
            positions[pos] = count / shots
        
        return positions

class QuantumMarketSimulator:
    """Quantum market simulator using quantum random walks"""
    
    def __init__(self, num_qubits=8):
        self.num_qubits = num_qubits
        self.qrw = QuantumRandomWalk(num_qubits)
        
    def create_market_state(self, initial_price=100.0, volatility=0.1):
        """
        Create quantum state representing market
        """
        # Normalize price to quantum state
        normalized_price = min(initial_price / 200.0, 1.0)  # Assume max price 200
        
        # Create quantum circuit
        circuit = QuantumCircuit(self.num_qubits + 1)
        
        # Encode initial price
        price_qubits = int(normalized_price * (2**self.num_qubits - 1))
        if price_qubits > 0:
            circuit.x(price_qubits)
        
        # Add volatility as superposition
        circuit.h(self.num_qubits)  # Coin qubit for volatility
        
        return circuit
    
    def simulate_market_evolution(self, initial_price=100.0, steps=50, volatility=0.1):
        """
        Simulate market evolution using quantum random walk
        """
        # Create initial market state
        circuit = self.create_market_state(initial_price, volatility)
        
        # Apply quantum walk steps
        for step in range(steps):
            circuit = self.qrw.quantum_walk_step(circuit, step)
        
        # Measure final state
        circuit.measure_all()
        
        # Execute circuit
        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Convert to price distribution
        price_distribution = self._extract_price_distribution(counts)
        
        return price_distribution, circuit
    
    def _extract_price_distribution(self, counts):
        """
        Extract price distribution from measurement counts
        """
        total_shots = sum(counts.values())
        prices = []
        
        for state, count in counts.items():
            # Convert quantum state to price
            state_value = int(state[:-1], 2)  # Exclude coin qubit
            price = (state_value / (2**self.num_qubits - 1)) * 200.0  # Scale back to price
            
            # Add price multiple times based on count
            prices.extend([price] * count)
        
        return np.array(prices)
    
    def calculate_market_metrics(self, price_distribution):
        """
        Calculate market metrics from price distribution
        """
        mean_price = np.mean(price_distribution)
        std_price = np.std(price_distribution)
        skewness = self._calculate_skewness(price_distribution)
        kurtosis = self._calculate_kurtosis(price_distribution)
        
        return {
            'mean': mean_price,
            'std': std_price,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _calculate_skewness(self, data):
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3

def compare_random_walks():
    """
    Compare classical and quantum random walks
    """
    print("=== Classical vs Quantum Random Walks ===\n")
    
    # Classical random walk
    print("1. Classical Random Walk:")
    crw = ClassicalRandomWalk(steps=100, drift=0.001, volatility=0.02)
    
    # Simulate multiple paths
    classical_paths = crw.simulate_multiple_paths(n_paths=1000)
    
    # Calculate classical statistics
    final_prices = classical_paths[:, -1]
    classical_mean = np.mean(final_prices)
    classical_std = np.std(final_prices)
    
    print(f"   Mean Final Price: {classical_mean:.2f}")
    print(f"   Std Final Price: {classical_std:.2f}")
    
    # Quantum random walk
    print("\n2. Quantum Random Walk:")
    qrw = QuantumRandomWalk(num_qubits=8)
    
    # Simulate quantum walk
    circuit = qrw.simulate_quantum_walk(steps=50)
    position_dist = qrw.get_position_distribution(circuit)
    
    # Convert positions to prices
    quantum_prices = position_dist * 200.0  # Scale to price range
    quantum_mean = np.mean(quantum_prices)
    quantum_std = np.std(quantum_prices)
    
    print(f"   Mean Final Price: {quantum_mean:.2f}")
    print(f"   Std Final Price: {quantum_std:.2f}")
    
    # Compare distributions
    print(f"\n3. Comparison:")
    print(f"   Classical Std/Mean: {classical_std/classical_mean:.4f}")
    print(f"   Quantum Std/Mean: {quantum_std/quantum_mean:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Classical paths
    plt.subplot(1, 3, 1)
    for i in range(min(100, len(classical_paths))):
        plt.plot(classical_paths[i], alpha=0.1, color='blue')
    plt.title('Classical Random Walk Paths')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    
    # Classical final distribution
    plt.subplot(1, 3, 2)
    plt.hist(final_prices, bins=50, alpha=0.7, color='blue', label='Classical')
    plt.title('Classical Final Price Distribution')
    plt.xlabel('Final Price')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Quantum distribution
    plt.subplot(1, 3, 3)
    plt.hist(quantum_prices, bins=50, alpha=0.7, color='orange', label='Quantum')
    plt.title('Quantum Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return classical_paths, quantum_prices

def quantum_market_simulation():
    """
    Quantum market simulation demo
    """
    print("=== Quantum Market Simulation ===\n")
    
    # Initialize quantum market simulator
    qms = QuantumMarketSimulator(num_qubits=8)
    
    # Simulate market evolution
    initial_price = 100.0
    steps = 50
    volatility = 0.1
    
    price_dist, circuit = qms.simulate_market_evolution(
        initial_price=initial_price,
        steps=steps,
        volatility=volatility
    )
    
    # Calculate market metrics
    metrics = qms.calculate_market_metrics(price_dist)
    
    print(f"Initial Price: ${initial_price:.2f}")
    print(f"Simulation Steps: {steps}")
    print(f"Volatility: {volatility}")
    print(f"\nMarket Metrics:")
    print(f"  Mean Price: ${metrics['mean']:.2f}")
    print(f"  Standard Deviation: ${metrics['std']:.2f}")
    print(f"  Skewness: {metrics['skewness']:.4f}")
    print(f"  Kurtosis: {metrics['kurtosis']:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Price distribution
    plt.subplot(1, 2, 1)
    plt.hist(price_dist, bins=50, alpha=0.7, color='green')
    plt.axvline(initial_price, color='red', linestyle='--', label='Initial Price')
    plt.axvline(metrics['mean'], color='blue', linestyle='--', label='Mean Price')
    plt.title('Quantum Market Price Distribution')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Price evolution (simplified)
    plt.subplot(1, 2, 2)
    # Create time series from distribution
    time_steps = np.linspace(0, steps, len(price_dist))
    plt.scatter(time_steps, price_dist, alpha=0.1, s=1)
    plt.plot([0, steps], [initial_price, initial_price], 'r--', label='Initial Price')
    plt.plot([0, steps], [metrics['mean'], metrics['mean']], 'b--', label='Mean Price')
    plt.title('Quantum Market Evolution')
    plt.xlabel('Time Step')
    plt.ylabel('Price ($)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return price_dist, metrics, circuit

# Exercise: Quantum Walk on Financial Networks
def quantum_walk_financial_network():
    """
    Exercise: Implement quantum walk on financial network
    """
    # Create simple financial network (banks connected by lending)
    num_banks = 4
    adjacency_matrix = np.array([
        [0, 1, 1, 0],  # Bank 0 lends to banks 1 and 2
        [1, 0, 1, 1],  # Bank 1 lends to banks 0, 2, and 3
        [1, 1, 0, 1],  # Bank 2 lends to banks 0, 1, and 3
        [0, 1, 1, 0]   # Bank 3 lends to banks 1 and 2
    ])
    
    # Create quantum circuit for network walk
    circuit = QuantumCircuit(num_banks, num_banks)
    
    # Initialize at bank 0
    circuit.x(0)
    
    # Apply quantum walk steps
    for step in range(10):
        # Hadamard on each qubit (coin flip)
        for i in range(num_banks):
            circuit.h(i)
        
        # Controlled operations based on adjacency matrix
        for i in range(num_banks):
            for j in range(num_banks):
                if adjacency_matrix[i, j] == 1 and i != j:
                    circuit.cx(i, j)
    
    # Measure
    circuit.measure_all()
    
    # Execute circuit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    # Analyze results
    print("=== Quantum Walk on Financial Network ===")
    print("Adjacency Matrix:")
    print(adjacency_matrix)
    print("\nBank Visit Probabilities:")
    
    bank_probs = np.zeros(num_banks)
    total_shots = sum(counts.values())
    
    for state, count in counts.items():
        # Find which bank is visited (has 1)
        for i, bit in enumerate(state):
            if bit == '1':
                bank_probs[i] += count / total_shots
    
    for i, prob in enumerate(bank_probs):
        print(f"  Bank {i}: {prob:.4f}")
    
    return circuit, counts, bank_probs

# Run demos
if __name__ == "__main__":
    print("Running Random Walk Comparisons...")
    classical_paths, quantum_prices = compare_random_walks()
    
    print("\nRunning Quantum Market Simulation...")
    price_dist, metrics, circuit = quantum_market_simulation()
    
    print("\nRunning Financial Network Exercise...")
    network_circuit, network_counts, bank_probs = quantum_walk_financial_network()
```

### **Exercise 2: Quantum Walk Optimization**

```python
def quantum_walk_optimization():
    """
    Exercise: Optimize quantum walk parameters for market modeling
    """
    from scipy.optimize import minimize
    
    def objective_function(params):
        """
        Objective function for quantum walk optimization
        """
        steps, coin_angle = params
        
        # Create quantum walk with parameters
        qrw = QuantumRandomWalk(num_qubits=6)
        circuit = qrw.simulate_quantum_walk(steps=int(steps))
        
        # Get distribution
        position_dist = qrw.get_position_distribution(circuit)
        
        # Calculate target metrics (e.g., match market volatility)
        target_volatility = 0.2
        actual_volatility = np.std(position_dist)
        
        # Return error
        return abs(actual_volatility - target_volatility)
    
    # Optimize parameters
    initial_params = [25, np.pi/4]  # Initial steps and coin angle
    bounds = [(10, 100), (0, np.pi)]  # Parameter bounds
    
    result = minimize(objective_function, initial_params, bounds=bounds)
    
    print("=== Quantum Walk Optimization ===")
    print(f"Optimal Steps: {int(result.x[0])}")
    print(f"Optimal Coin Angle: {result.x[1]:.4f}")
    print(f"Optimization Error: {result.fun:.6f}")
    
    return result

def quantum_walk_entanglement_analysis():
    """
    Exercise: Analyze entanglement in quantum walks
    """
    from qiskit.quantum_info import entanglement_of_formation
    
    # Create quantum walk with different initial states
    qrw = QuantumRandomWalk(num_qubits=4)
    
    # Different initial states
    initial_states = [
        "separable",  # |0000⟩
        "entangled",  # Bell state
        "mixed"       # Mixed state
    ]
    
    entanglement_measures = []
    
    for state_type in initial_states:
        # Create circuit with different initial states
        circuit = QuantumCircuit(5)  # 4 position + 1 coin
        
        if state_type == "separable":
            circuit.x(0)  # Start at position 0
        elif state_type == "entangled":
            circuit.h(0)
            circuit.cx(0, 1)  # Create Bell state
        elif state_type == "mixed":
            circuit.h(0)
            circuit.h(1)
        
        # Add coin
        circuit.h(4)
        
        # Apply quantum walk steps
        for step in range(10):
            circuit = qrw.quantum_walk_step(circuit, step)
        
        # Calculate entanglement
        # Note: This is a simplified calculation
        entanglement = calculate_simplified_entanglement(circuit)
        entanglement_measures.append(entanglement)
        
        print(f"{state_type.capitalize()} State Entanglement: {entanglement:.4f}")
    
    return entanglement_measures

def calculate_simplified_entanglement(circuit):
    """
    Simplified entanglement calculation
    """
    # Get statevector
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    statevector = result.get_statevector()
    
    # Calculate von Neumann entropy of reduced density matrix
    # This is a simplified measure of entanglement
    state_array = np.array(statevector)
    
    # Reshape to 2x2^(n-1) matrix
    n_qubits = circuit.num_qubits
    state_matrix = state_array.reshape(2, 2**(n_qubits-1))
    
    # Calculate reduced density matrix
    rho = state_matrix @ state_matrix.conj().T
    
    # Calculate von Neumann entropy
    eigenvals = np.linalg.eigvalsh(rho)
    eigenvals = eigenvals[eigenvals > 0]  # Remove zero eigenvalues
    entropy = -np.sum(eigenvals * np.log2(eigenvals))
    
    return entropy

# Run exercises
if __name__ == "__main__":
    print("Running Quantum Walk Optimization...")
    opt_result = quantum_walk_optimization()
    
    print("\nRunning Entanglement Analysis...")
    entanglement_measures = quantum_walk_entanglement_analysis()
```

## 📊 Kết quả và Phân tích

### **Quantum Random Walk Advantages:**

#### **1. Speedup:**
- **Quadratic Speedup**: Cho certain search problems
- **Exponential Speedup**: Cho specific graph problems
- **Parallel Processing**: Superposition of multiple paths

#### **2. Distribution Properties:**
- **Non-Gaussian**: Captures fat tails
- **Ballistic Spreading**: Faster than classical diffusion
- **Quantum Interference**: Complex correlation patterns

#### **3. Market Applications:**
- **Price Evolution**: Quantum price dynamics
- **Volatility Modeling**: Quantum volatility measures
- **Network Effects**: Quantum walks on financial networks

### **Comparison với Classical Random Walks:**

#### **Classical Limitations:**
- Linear variance growth
- Gaussian distributions
- No interference effects

#### **Quantum Advantages:**
- Quadratic variance growth
- Non-Gaussian distributions
- Quantum interference

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Walk Calibration**
Implement quantum walk calibration cho market data.

### **Exercise 2: Quantum Walk Networks**
Build quantum walks on complex financial networks.

### **Exercise 3: Quantum Walk Optimization**
Develop optimization algorithms cho quantum walk parameters.

### **Exercise 4: Quantum Walk Validation**
Create validation framework cho quantum walk models.

---

> *"Quantum random walks provide a more sophisticated model of market dynamics, capturing the quantum nature of financial information flow."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Monte Carlo Methods](Day8.md) 