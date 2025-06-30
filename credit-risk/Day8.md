# Ngày 8: Quantum Monte Carlo Methods

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum Monte Carlo methods và classical Monte Carlo
- Nắm vững cách quantum Monte Carlo cải thiện financial simulations
- Implement quantum-enhanced Monte Carlo cho credit risk modeling
- So sánh performance giữa quantum và classical Monte Carlo

## 📚 Lý thuyết

### **Monte Carlo Methods Fundamentals**

#### **1. Classical Monte Carlo**

**Basic Principle:**
```
E[f(X)] ≈ (1/N) Σᵢ f(xᵢ)
xᵢ ~ P(X)
```

**Applications in Finance:**
- Option pricing
- Risk measures (VaR, CVaR)
- Portfolio optimization
- Credit risk assessment

#### **2. Quantum Monte Carlo**

**Quantum Advantage:**
- True quantum randomness
- Parallel scenario processing
- Quantum speedup cho certain problems

**Quantum State Representation:**
```
|ψ⟩ = Σₓ √p(x)|x⟩
```

### **Quantum Monte Carlo Types**

#### **1. Quantum Amplitude Estimation:**

**Classical vs Quantum:**
```
Classical: O(1/ε²) samples
Quantum: O(1/ε) samples
```

**Algorithm:**
```
|0⟩ → H → U → QFT → Measure
```

#### **2. Quantum Phase Estimation:**

**Principle:**
```
U|ψ⟩ = e^(2πiφ)|ψ⟩
```

**Application:**
- Eigenvalue estimation
- Financial derivative pricing

#### **3. Quantum Variational Monte Carlo:**

**Variational Principle:**
```
E[ψ] = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩
```

**Optimization:**
```
min E[ψ(θ)]
```

### **Financial Applications**

#### **1. Credit Risk Modeling:**

**Default Probability:**
```
PD = E[1_{default}]
```

**Expected Loss:**
```
EL = E[LGD × EAD × 1_{default}]
```

#### **2. Portfolio Risk:**

**Value at Risk:**
```
VaR_α = inf{V : P(L > V) ≤ α}
```

**Conditional VaR:**
```
CVaR_α = E[L | L > VaR_α]
```

#### **3. Option Pricing:**

**European Option:**
```
C = E[e^(-rT) max(S_T - K, 0)]
```

## 💻 Thực hành

### **Project 8: Quantum Monte Carlo Credit Risk Simulator**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import QFT, RealAmplitudes
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit_finance.applications.optimization import PortfolioOptimization
import pennylane as qml

class ClassicalMonteCarlo:
    """Classical Monte Carlo implementation"""
    
    def __init__(self, n_simulations=10000):
        self.n_simulations = n_simulations
        
    def simulate_default_probability(self, credit_scores, default_threshold=0.5):
        """
        Simulate default probability using classical Monte Carlo
        """
        defaults = 0
        
        for _ in range(self.n_simulations):
            # Generate random default event
            default_prob = np.random.uniform(0, 1)
            
            # Check if default occurs
            if default_prob < default_threshold:
                defaults += 1
        
        return defaults / self.n_simulations
    
    def simulate_portfolio_loss(self, portfolio_weights, asset_returns, default_probs):
        """
        Simulate portfolio loss using classical Monte Carlo
        """
        losses = []
        
        for _ in range(self.n_simulations):
            # Simulate default events
            defaults = np.random.binomial(1, default_probs)
            
            # Calculate portfolio loss
            loss = np.sum(portfolio_weights * asset_returns * defaults)
            losses.append(loss)
        
        return np.array(losses)
    
    def calculate_var_cvar(self, losses, alpha=0.05):
        """
        Calculate VaR and CVaR from loss distribution
        """
        sorted_losses = np.sort(losses)
        var_index = int(alpha * len(sorted_losses))
        var = sorted_losses[var_index]
        
        # CVaR is expected loss beyond VaR
        cvar_losses = sorted_losses[var_index:]
        cvar = np.mean(cvar_losses)
        
        return var, cvar

class QuantumMonteCarlo:
    """Quantum Monte Carlo implementation"""
    
    def __init__(self, num_qubits=8):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_quantum_random_generator(self):
        """
        Create quantum circuit for random number generation
        """
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Apply Hadamard gates to create superposition
        for i in range(self.num_qubits):
            circuit.h(i)
        
        # Add some entanglement for better randomness
        for i in range(self.num_qubits - 1):
            circuit.cx(i, i + 1)
        
        # Measure
        circuit.measure_all()
        
        return circuit
    
    def generate_quantum_random_numbers(self, n_numbers=1000):
        """
        Generate random numbers using quantum circuit
        """
        circuit = self.create_quantum_random_generator()
        
        # Execute multiple times
        job = execute(circuit, self.backend, shots=n_numbers)
        result = job.result()
        counts = result.get_counts()
        
        # Convert to random numbers
        random_numbers = []
        for state, count in counts.items():
            # Convert binary to decimal
            decimal_value = int(state, 2) / (2**self.num_qubits - 1)
            random_numbers.extend([decimal_value] * count)
        
        return np.array(random_numbers)
    
    def quantum_amplitude_estimation(self, target_probability, precision=0.01):
        """
        Implement quantum amplitude estimation
        """
        # Create quantum circuit for amplitude estimation
        n_estimation_qubits = int(np.log2(1/precision))
        total_qubits = n_estimation_qubits + 1
        
        circuit = QuantumCircuit(total_qubits, n_estimation_qubits)
        
        # Initialize target state
        circuit.x(0)
        
        # Apply Hadamard to estimation qubits
        for i in range(1, total_qubits):
            circuit.h(i)
        
        # Apply controlled operations
        for i in range(n_estimation_qubits):
            # Apply rotation based on target probability
            angle = 2 * np.arccos(np.sqrt(target_probability))
            circuit.cry(angle, i + 1, 0)
        
        # Apply inverse QFT
        circuit.h(1)
        for i in range(2, total_qubits):
            circuit.cp(np.pi / (2**(i-1)), i, i-1)
            circuit.h(i)
        
        # Measure estimation qubits
        circuit.measure(list(range(1, total_qubits)), list(range(n_estimation_qubits)))
        
        return circuit
    
    def estimate_probability_quantum(self, target_prob, shots=1000):
        """
        Estimate probability using quantum amplitude estimation
        """
        circuit = self.quantum_amplitude_estimation(target_prob)
        
        # Execute circuit
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Extract probability estimate
        total_shots = sum(counts.values())
        probability_estimate = 0
        
        for state, count in counts.items():
            # Convert measurement to probability estimate
            state_value = int(state, 2)
            normalized_value = state_value / (2**(len(state)) - 1)
            probability_estimate += (normalized_value * count) / total_shots
        
        return probability_estimate

class QuantumCreditRiskSimulator:
    """Quantum-enhanced credit risk simulator"""
    
    def __init__(self, num_qubits=8):
        self.num_qubits = num_qubits
        self.qmc = QuantumMonteCarlo(num_qubits)
        
    def simulate_quantum_default_probability(self, credit_data, n_simulations=1000):
        """
        Simulate default probability using quantum Monte Carlo
        """
        default_probs = []
        
        for _ in range(n_simulations):
            # Generate quantum random numbers
            random_numbers = self.qmc.generate_quantum_random_numbers(1)
            
            # Calculate default probability based on credit data
            credit_score = np.random.choice(credit_data['credit_score'])
            default_threshold = 1 - (credit_score / 850)  # Normalize to [0,1]
            
            # Use quantum random number for default decision
            default_prob = 1 if random_numbers[0] < default_threshold else 0
            default_probs.append(default_prob)
        
        return np.mean(default_probs)
    
    def simulate_quantum_portfolio_loss(self, portfolio_data, n_simulations=1000):
        """
        Simulate portfolio loss using quantum Monte Carlo
        """
        losses = []
        
        for _ in range(n_simulations):
            # Generate quantum random numbers for each asset
            n_assets = len(portfolio_data['weights'])
            random_numbers = self.qmc.generate_quantum_random_numbers(n_assets)
            
            # Simulate defaults using quantum randomness
            defaults = (random_numbers < portfolio_data['default_probs']).astype(int)
            
            # Calculate portfolio loss
            loss = np.sum(portfolio_data['weights'] * 
                         portfolio_data['exposures'] * 
                         portfolio_data['lgd'] * defaults)
            
            losses.append(loss)
        
        return np.array(losses)
    
    def calculate_quantum_var_cvar(self, losses, alpha=0.05):
        """
        Calculate VaR and CVaR using quantum methods
        """
        # Sort losses
        sorted_losses = np.sort(losses)
        
        # Use quantum amplitude estimation for percentile
        var_index = int(alpha * len(sorted_losses))
        var = sorted_losses[var_index]
        
        # Calculate CVaR
        cvar_losses = sorted_losses[var_index:]
        cvar = np.mean(cvar_losses)
        
        return var, cvar

def compare_monte_carlo_methods():
    """
    Compare classical and quantum Monte Carlo methods
    """
    print("=== Classical vs Quantum Monte Carlo Comparison ===\n")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Credit data
    credit_scores = np.random.normal(700, 100, n_samples)
    default_probs = 1 - (credit_scores / 850)
    
    # Portfolio data
    portfolio_data = {
        'weights': [0.25, 0.25, 0.25, 0.25],
        'exposures': [1000000, 1500000, 800000, 1200000],
        'lgd': [0.4, 0.5, 0.3, 0.45],
        'default_probs': [0.02, 0.03, 0.01, 0.025]
    }
    
    # Classical Monte Carlo
    print("1. Classical Monte Carlo:")
    cmc = ClassicalMonteCarlo(n_simulations=10000)
    
    # Simulate default probability
    classical_default_prob = cmc.simulate_default_probability(credit_scores)
    print(f"   Default Probability: {classical_default_prob:.4f}")
    
    # Simulate portfolio loss
    classical_losses = cmc.simulate_portfolio_loss(
        portfolio_data['weights'],
        portfolio_data['exposures'],
        portfolio_data['default_probs']
    )
    
    # Calculate VaR and CVaR
    classical_var, classical_cvar = cmc.calculate_var_cvar(classical_losses)
    print(f"   VaR (95%): ${classical_var:,.2f}")
    print(f"   CVaR (95%): ${classical_cvar:,.2f}")
    
    # Quantum Monte Carlo
    print("\n2. Quantum Monte Carlo:")
    qcrs = QuantumCreditRiskSimulator(num_qubits=8)
    
    # Simulate default probability
    quantum_default_prob = qcrs.simulate_quantum_default_probability(
        {'credit_score': credit_scores}
    )
    print(f"   Default Probability: {quantum_default_prob:.4f}")
    
    # Simulate portfolio loss
    quantum_losses = qcrs.simulate_quantum_portfolio_loss(portfolio_data)
    
    # Calculate VaR and CVaR
    quantum_var, quantum_cvar = qcrs.calculate_quantum_var_cvar(quantum_losses)
    print(f"   VaR (95%): ${quantum_var:,.2f}")
    print(f"   CVaR (95%): ${quantum_cvar:,.2f}")
    
    # Compare results
    print(f"\n3. Comparison:")
    print(f"   Default Prob Difference: {abs(classical_default_prob - quantum_default_prob):.4f}")
    print(f"   VaR Difference: ${abs(classical_var - quantum_var):,.2f}")
    print(f"   CVaR Difference: ${abs(classical_cvar - quantum_cvar):,.2f}")
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Default probability comparison
    plt.subplot(1, 3, 1)
    methods = ['Classical', 'Quantum']
    probs = [classical_default_prob, quantum_default_prob]
    plt.bar(methods, probs, color=['blue', 'orange'])
    plt.title('Default Probability Comparison')
    plt.ylabel('Probability')
    
    # Loss distribution comparison
    plt.subplot(1, 3, 2)
    plt.hist(classical_losses, bins=50, alpha=0.7, label='Classical', color='blue')
    plt.hist(quantum_losses, bins=50, alpha=0.7, label='Quantum', color='orange')
    plt.title('Portfolio Loss Distribution')
    plt.xlabel('Loss ($)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # VaR/CVaR comparison
    plt.subplot(1, 3, 3)
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, [classical_var, classical_cvar], width, label='Classical', color='blue')
    plt.bar(x + width/2, [quantum_var, quantum_cvar], width, label='Quantum', color='orange')
    plt.xlabel('Risk Measure')
    plt.ylabel('Value ($)')
    plt.title('VaR/CVaR Comparison')
    plt.xticks(x, ['VaR', 'CVaR'])
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return classical_losses, quantum_losses

def quantum_amplitude_estimation_demo():
    """
    Demo quantum amplitude estimation
    """
    print("=== Quantum Amplitude Estimation Demo ===\n")
    
    # Initialize quantum Monte Carlo
    qmc = QuantumMonteCarlo(num_qubits=8)
    
    # Test different target probabilities
    target_probs = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    print("Target Probability | Quantum Estimate | Error")
    print("-" * 45)
    
    for target_prob in target_probs:
        # Estimate using quantum amplitude estimation
        quantum_estimate = qmc.estimate_probability_quantum(target_prob)
        error = abs(target_prob - quantum_estimate)
        
        print(f"{target_prob:16.2f} | {quantum_estimate:15.4f} | {error:5.4f}")
    
    return target_probs

def quantum_variational_monte_carlo():
    """
    Implement quantum variational Monte Carlo
    """
    print("=== Quantum Variational Monte Carlo ===\n")
    
    # Create variational quantum circuit
    num_qubits = 4
    circuit = QuantumCircuit(num_qubits)
    
    # Add variational parameters
    params = np.random.uniform(0, 2*np.pi, num_qubits)
    
    # Apply parameterized rotations
    for i in range(num_qubits):
        circuit.rx(params[i], i)
    
    # Add entanglement
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
    
    # Define cost function (example: energy expectation)
    def cost_function(params):
        """
        Cost function for variational optimization
        """
        # Create circuit with parameters
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.rx(params[i], i)
        
        # Add entanglement
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        
        # Calculate energy expectation (simplified)
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        result = job.result()
        statevector = result.get_statevector()
        
        # Simple energy calculation
        energy = np.real(np.sum(statevector * np.conj(statevector) * np.arange(len(statevector))))
        return energy
    
    # Optimize parameters
    from scipy.optimize import minimize
    
    result = minimize(cost_function, params, method='L-BFGS-B')
    
    print(f"Optimization Result:")
    print(f"  Optimal Energy: {result.fun:.4f}")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.nit}")
    
    return circuit, result

# Exercise: Quantum Monte Carlo for Option Pricing
def quantum_option_pricing():
    """
    Exercise: Implement quantum Monte Carlo for option pricing
    """
    # European call option parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    T = 1.0     # Time to maturity
    r = 0.05    # Risk-free rate
    sigma = 0.2 # Volatility
    
    def classical_option_price(n_simulations=10000):
        """
        Classical Monte Carlo for option pricing
        """
        payoffs = []
        
        for _ in range(n_simulations):
            # Generate stock price path
            Z = np.random.normal(0, 1)
            ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
            
            # Calculate payoff
            payoff = max(ST - K, 0)
            payoffs.append(payoff)
        
        # Discount expected payoff
        option_price = np.exp(-r * T) * np.mean(payoffs)
        return option_price
    
    def quantum_option_price(n_simulations=1000):
        """
        Quantum Monte Carlo for option pricing
        """
        qmc = QuantumMonteCarlo(num_qubits=8)
        payoffs = []
        
        for _ in range(n_simulations):
            # Generate quantum random numbers
            random_numbers = qmc.generate_quantum_random_numbers(1)
            
            # Convert to normal distribution using Box-Muller
            Z = np.sqrt(-2 * np.log(random_numbers[0])) * np.cos(2 * np.pi * random_numbers[0])
            
            # Generate stock price
            ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
            
            # Calculate payoff
            payoff = max(ST - K, 0)
            payoffs.append(payoff)
        
        # Discount expected payoff
        option_price = np.exp(-r * T) * np.mean(payoffs)
        return option_price
    
    # Calculate option prices
    classical_price = classical_option_price()
    quantum_price = quantum_option_price()
    
    # Black-Scholes analytical price for comparison
    from scipy.stats import norm
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    analytical_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    print("=== Quantum Option Pricing ===")
    print(f"Analytical Price: ${analytical_price:.4f}")
    print(f"Classical MC Price: ${classical_price:.4f}")
    print(f"Quantum MC Price: ${quantum_price:.4f}")
    print(f"Classical Error: ${abs(classical_price - analytical_price):.4f}")
    print(f"Quantum Error: ${abs(quantum_price - analytical_price):.4f}")
    
    return classical_price, quantum_price, analytical_price

# Run demos
if __name__ == "__main__":
    print("Running Monte Carlo Comparisons...")
    classical_losses, quantum_losses = compare_monte_carlo_methods()
    
    print("\nRunning Amplitude Estimation Demo...")
    target_probs = quantum_amplitude_estimation_demo()
    
    print("\nRunning Variational Monte Carlo...")
    circuit, result = quantum_variational_monte_carlo()
    
    print("\nRunning Option Pricing Exercise...")
    classical_price, quantum_price, analytical_price = quantum_option_pricing()
```

### **Exercise 2: Quantum Monte Carlo Optimization**

```python
def quantum_monte_carlo_optimization():
    """
    Exercise: Optimize quantum Monte Carlo parameters
    """
    from scipy.optimize import minimize
    
    def objective_function(params):
        """
        Objective function for quantum Monte Carlo optimization
        """
        n_qubits, n_shots = int(params[0]), int(params[1])
        
        # Create quantum Monte Carlo with parameters
        qmc = QuantumMonteCarlo(num_qubits=n_qubits)
        
        # Test on known probability
        target_prob = 0.5
        estimated_prob = qmc.estimate_probability_quantum(target_prob, shots=n_shots)
        
        # Return error
        return abs(target_prob - estimated_prob)
    
    # Optimize parameters
    initial_params = [8, 1000]  # Initial n_qubits, n_shots
    bounds = [(4, 12), (100, 10000)]  # Parameter bounds
    
    result = minimize(objective_function, initial_params, bounds=bounds)
    
    print("=== Quantum Monte Carlo Optimization ===")
    print(f"Optimal Number of Qubits: {int(result.x[0])}")
    print(f"Optimal Number of Shots: {int(result.x[1])}")
    print(f"Optimization Error: {result.fun:.6f}")
    
    return result

def quantum_monte_carlo_convergence():
    """
    Exercise: Analyze convergence of quantum Monte Carlo
    """
    # Test convergence with different numbers of shots
    shot_counts = [100, 500, 1000, 5000, 10000]
    target_prob = 0.3
    
    qmc = QuantumMonteCarlo(num_qubits=8)
    
    errors = []
    
    for shots in shot_counts:
        estimated_prob = qmc.estimate_probability_quantum(target_prob, shots=shots)
        error = abs(target_prob - estimated_prob)
        errors.append(error)
        
        print(f"Shots: {shots:5d} | Estimate: {estimated_prob:.4f} | Error: {error:.4f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(shot_counts, errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Shots')
    plt.ylabel('Absolute Error')
    plt.title('Quantum Monte Carlo Convergence')
    plt.grid(True)
    plt.yscale('log')
    plt.show()
    
    return shot_counts, errors

# Run exercises
if __name__ == "__main__":
    print("Running Quantum Monte Carlo Optimization...")
    opt_result = quantum_monte_carlo_optimization()
    
    print("\nRunning Convergence Analysis...")
    shot_counts, errors = quantum_monte_carlo_convergence()
```

## 📊 Kết quả và Phân tích

### **Quantum Monte Carlo Advantages:**

#### **1. Speedup:**
- **Amplitude Estimation**: Quadratic speedup cho probability estimation
- **Phase Estimation**: Exponential speedup cho certain problems
- **Parallel Processing**: Superposition of multiple scenarios

#### **2. Accuracy:**
- **True Randomness**: Quantum randomness vs pseudo-randomness
- **Quantum Interference**: Better sampling distributions
- **Variational Methods**: Adaptive parameter optimization

#### **3. Financial Applications:**
- **Credit Risk**: Quantum default probability estimation
- **Portfolio Risk**: Quantum VaR/CVaR calculation
- **Option Pricing**: Quantum derivative pricing

### **Comparison với Classical Monte Carlo:**

#### **Classical Limitations:**
- Pseudo-random number generation
- Sequential processing
- Limited by computational resources

#### **Quantum Advantages:**
- True quantum randomness
- Parallel scenario processing
- Quantum speedup cho certain problems

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Monte Carlo Calibration**
Implement quantum Monte Carlo calibration cho financial models.

### **Exercise 2: Quantum Monte Carlo Networks**
Build quantum Monte Carlo cho network-based risk models.

### **Exercise 3: Quantum Monte Carlo Optimization**
Develop optimization algorithms cho quantum Monte Carlo parameters.

### **Exercise 4: Quantum Monte Carlo Validation**
Create validation framework cho quantum Monte Carlo models.

---

> *"Quantum Monte Carlo methods provide exponential speedup for certain financial simulations, enabling real-time risk assessment."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Optimization cho Portfolio](Day9.md) 