# Ngày 16: Quantum Portfolio Optimization

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum portfolio optimization và classical portfolio optimization
- Nắm vững cách quantum computing cải thiện portfolio optimization
- Implement quantum portfolio optimization algorithms cho credit risk
- So sánh performance giữa quantum và classical portfolio optimization

## 📚 Lý thuyết

### **Portfolio Optimization Fundamentals**

#### **1. Classical Portfolio Optimization**

**Markowitz Model:**
```
min w^T Σ w
s.t. w^T μ = r_target
     w^T 1 = 1
     w ≥ 0
```

**Risk Measures:**
- **Variance**: `σ² = w^T Σ w`
- **VaR**: `P(L > VaR) = α`
- **CVaR**: `E[L|L > VaR]`

#### **2. Quantum Portfolio Optimization**

**Quantum State Representation:**
```
|ψ⟩ = Σᵢ wᵢ|i⟩
```

**Quantum Risk Function:**
```
R_quantum = ⟨ψ|H_risk|ψ⟩
```

**Quantum Constraint:**
```
⟨ψ|H_constraint|ψ⟩ = 0
```

### **Quantum Portfolio Optimization Types**

#### **1. Quantum Approximate Optimization Algorithm (QAOA):**
- **Problem Encoding**: Portfolio weights as qubits
- **Cost Function**: Risk-return objective
- **Constraints**: Budget and target return constraints

#### **2. Variational Quantum Eigensolver (VQE):**
- **Hamiltonian**: Risk matrix encoding
- **Ansatz**: Parameterized quantum circuit
- **Optimization**: Classical optimizer

#### **3. Quantum Annealing:**
- **QUBO Formulation**: Portfolio optimization as QUBO
- **Quantum Annealer**: D-Wave systems
- **Solution**: Optimal portfolio weights

### **Quantum Portfolio Optimization Advantages**

#### **1. Quantum Properties:**
- **Superposition**: Parallel evaluation of multiple portfolios
- **Entanglement**: Complex asset correlations
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Credit-specific Benefits:**
- **Non-linear Risk Models**: Quantum circuits capture complex risk relationships
- **High-dimensional Optimization**: Handle many assets efficiently
- **Quantum Advantage**: Potential speedup for large portfolios

## 💻 Thực hành

### **Project 16: Quantum Portfolio Optimization cho Credit Risk**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.opflow import PauliSumOp, StateFn, CircuitSampler
from qiskit.quantum_info import Pauli
import pennylane as qml

class ClassicalPortfolioOptimizer:
    """Classical portfolio optimization methods"""
    
    def __init__(self):
        self.weights = None
        self.expected_return = None
        self.risk = None
        
    def generate_credit_assets(self, n_assets=10, n_periods=252):
        """
        Generate synthetic credit asset returns
        """
        np.random.seed(42)
        
        # Generate asset returns with credit risk characteristics
        returns = np.random.multivariate_normal(
            mean=np.random.uniform(0.05, 0.15, n_assets),  # Expected returns
            cov=np.random.uniform(0.1, 0.3, (n_assets, n_assets)),  # Covariance
            size=n_periods
        )
        
        # Add credit risk events (defaults)
        for i in range(n_periods):
            if np.random.random() < 0.01:  # 1% default probability
                default_asset = np.random.randint(0, n_assets)
                returns[i, default_asset] = -0.5  # 50% loss on default
        
        return pd.DataFrame(returns, columns=[f'Asset_{i}' for i in range(n_assets)])
    
    def calculate_statistics(self, returns):
        """
        Calculate portfolio statistics
        """
        # Expected returns
        expected_returns = returns.mean()
        
        # Covariance matrix
        cov_matrix = returns.cov()
        
        # Risk-free rate
        risk_free_rate = 0.02
        
        return expected_returns, cov_matrix, risk_free_rate
    
    def markowitz_optimization(self, expected_returns, cov_matrix, target_return=None):
        """
        Classical Markowitz optimization
        """
        n_assets = len(expected_returns)
        
        # Objective function: minimize risk
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Budget constraint
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.dot(x, expected_returns) - target_return
            })
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            self.weights = result.x
            self.expected_return = np.dot(self.weights, expected_returns)
            self.risk = np.sqrt(np.dot(self.weights.T, np.dot(cov_matrix, self.weights)))
        
        return result.success
    
    def efficient_frontier(self, expected_returns, cov_matrix, n_points=50):
        """
        Generate efficient frontier
        """
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            success = self.markowitz_optimization(expected_returns, cov_matrix, target_return)
            if success:
                efficient_portfolios.append({
                    'return': self.expected_return,
                    'risk': self.risk,
                    'weights': self.weights.copy()
                })
        
        return efficient_portfolios

class QuantumPortfolioOptimizer:
    """Quantum portfolio optimization implementation"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.optimizer = SPSA(maxiter=100)
        self.weights = None
        self.expected_return = None
        self.risk = None
        
    def create_portfolio_circuit(self, expected_returns, cov_matrix):
        """
        Create quantum circuit for portfolio optimization
        """
        # Feature map for encoding returns and covariance
        feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=2)
        
        # Ansatz for parameterized portfolio weights
        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=3)
        
        # Combine feature map and ansatz
        circuit = feature_map.compose(ansatz)
        
        return circuit
    
    def create_risk_hamiltonian(self, cov_matrix):
        """
        Create risk Hamiltonian from covariance matrix
        """
        # Encode covariance matrix as Pauli operators
        n_assets = min(self.num_qubits, len(cov_matrix))
        
        # Create risk Hamiltonian (simplified)
        risk_terms = []
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    # Diagonal terms (variance)
                    pauli = Pauli('I' * i + 'Z' + 'I' * (n_assets - i - 1))
                    coeff = cov_matrix.iloc[i, i] / 4
                    risk_terms.append((coeff, pauli))
                else:
                    # Off-diagonal terms (covariance)
                    pauli_i = Pauli('I' * i + 'Z' + 'I' * (n_assets - i - 1))
                    pauli_j = Pauli('I' * j + 'Z' + 'I' * (n_assets - j - 1))
                    coeff = cov_matrix.iloc[i, j] / 4
                    risk_terms.append((coeff, pauli_i @ pauli_j))
        
        return PauliSumOp.from_list(risk_terms)
    
    def create_return_hamiltonian(self, expected_returns):
        """
        Create return Hamiltonian from expected returns
        """
        n_assets = min(self.num_qubits, len(expected_returns))
        
        return_terms = []
        for i in range(n_assets):
            pauli = Pauli('I' * i + 'Z' + 'I' * (n_assets - i - 1))
            coeff = expected_returns.iloc[i] / 2
            return_terms.append((coeff, pauli))
        
        return PauliSumOp.from_list(return_terms)
    
    def quantum_portfolio_objective(self, parameters, expected_returns, cov_matrix, 
                                   target_return=None, risk_weight=1.0):
        """
        Quantum portfolio optimization objective function
        """
        # Create quantum circuit
        circuit = self.create_portfolio_circuit(expected_returns, cov_matrix)
        
        # Bind parameters
        bound_circuit = circuit.bind_parameters(parameters)
        
        # Execute circuit
        job = execute(bound_circuit, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate portfolio weights from quantum state
        weights = self._extract_weights_from_counts(counts)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate portfolio statistics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Objective: minimize risk (with return constraint)
        objective = risk_weight * portfolio_risk
        
        if target_return is not None:
            # Add penalty for not meeting target return
            return_penalty = 1000 * max(0, target_return - portfolio_return)
            objective += return_penalty
        
        # Add penalty for budget constraint violation
        budget_penalty = 1000 * abs(np.sum(weights) - 1)
        objective += budget_penalty
        
        return objective
    
    def _extract_weights_from_counts(self, counts):
        """
        Extract portfolio weights from quantum measurement counts
        """
        total_shots = sum(counts.values())
        weights = np.zeros(self.num_qubits)
        
        for bitstring, count in counts.items():
            probability = count / total_shots
            
            # Convert bitstring to weights
            for i, bit in enumerate(bitstring):
                if i < self.num_qubits:
                    weights[i] += probability * (1 if bit == '1' else 0)
        
        return weights
    
    def optimize_portfolio(self, expected_returns, cov_matrix, target_return=None):
        """
        Optimize portfolio using quantum algorithm
        """
        # Initialize parameters
        circuit = self.create_portfolio_circuit(expected_returns, cov_matrix)
        initial_params = np.random.random(circuit.num_parameters) * 2 * np.pi
        
        # Optimize
        result = self.optimizer.minimize(
            fun=lambda params: self.quantum_portfolio_objective(
                params, expected_returns, cov_matrix, target_return
            ),
            x0=initial_params
        )
        
        if result.success:
            # Extract final weights
            final_circuit = circuit.bind_parameters(result.x)
            job = execute(final_circuit, self.backend, shots=1000)
            result_counts = job.result().get_counts()
            
            self.weights = self._extract_weights_from_counts(result_counts)
            self.weights = self.weights / np.sum(self.weights)  # Normalize
            
            self.expected_return = np.dot(self.weights, expected_returns)
            self.risk = np.sqrt(np.dot(self.weights.T, np.dot(cov_matrix, self.weights)))
        
        return result.success
    
    def quantum_efficient_frontier(self, expected_returns, cov_matrix, n_points=20):
        """
        Generate quantum efficient frontier
        """
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        quantum_portfolios = []
        
        for target_return in target_returns:
            print(f"Optimizing for target return: {target_return:.4f}")
            success = self.optimize_portfolio(expected_returns, cov_matrix, target_return)
            if success:
                quantum_portfolios.append({
                    'return': self.expected_return,
                    'risk': self.risk,
                    'weights': self.weights.copy()
                })
        
        return quantum_portfolios

def compare_portfolio_optimization():
    """
    Compare classical and quantum portfolio optimization
    """
    print("=== Classical vs Quantum Portfolio Optimization ===\n")
    
    # Generate credit asset data
    classical_optimizer = ClassicalPortfolioOptimizer()
    returns_data = classical_optimizer.generate_credit_assets(n_assets=8, n_periods=252)
    
    # Calculate statistics
    expected_returns, cov_matrix, risk_free_rate = classical_optimizer.calculate_statistics(returns_data)
    
    print("1. Classical Portfolio Optimization:")
    
    # Classical efficient frontier
    classical_frontier = classical_optimizer.efficient_frontier(expected_returns, cov_matrix, n_points=20)
    
    print(f"   Number of efficient portfolios: {len(classical_frontier)}")
    if classical_frontier:
        classical_risks = [p['risk'] for p in classical_frontier]
        classical_returns = [p['return'] for p in classical_frontier]
        print(f"   Risk range: [{min(classical_risks):.4f}, {max(classical_risks):.4f}]")
        print(f"   Return range: [{min(classical_returns):.4f}, {max(classical_returns):.4f}]")
    
    print("\n2. Quantum Portfolio Optimization:")
    
    # Quantum portfolio optimization
    quantum_optimizer = QuantumPortfolioOptimizer(num_qubits=4)
    
    # Use subset of assets for quantum optimization
    subset_returns = expected_returns[:4]
    subset_cov = cov_matrix.iloc[:4, :4]
    
    quantum_frontier = quantum_optimizer.quantum_efficient_frontier(
        subset_returns, subset_cov, n_points=10
    )
    
    print(f"   Number of quantum portfolios: {len(quantum_frontier)}")
    if quantum_frontier:
        quantum_risks = [p['risk'] for p in quantum_frontier]
        quantum_returns = [p['return'] for p in quantum_frontier]
        print(f"   Risk range: [{min(quantum_risks):.4f}, {max(quantum_risks):.4f}]")
        print(f"   Return range: [{min(quantum_returns):.4f}, {max(quantum_returns):.4f}]")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Efficient frontier comparison
    plt.subplot(1, 3, 1)
    if classical_frontier:
        classical_risks = [p['risk'] for p in classical_frontier]
        classical_returns = [p['return'] for p in classical_frontier]
        plt.plot(classical_risks, classical_returns, 'b-', label='Classical', linewidth=2)
    
    if quantum_frontier:
        quantum_risks = [p['risk'] for p in quantum_frontier]
        quantum_returns = [p['return'] for p in quantum_frontier]
        plt.plot(quantum_risks, quantum_returns, 'r--', label='Quantum', linewidth=2)
    
    plt.xlabel('Portfolio Risk (Volatility)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier Comparison')
    plt.legend()
    plt.grid(True)
    
    # Risk-return scatter plot
    plt.subplot(1, 3, 2)
    if classical_frontier:
        plt.scatter(classical_risks, classical_returns, c='blue', alpha=0.7, label='Classical')
    if quantum_frontier:
        plt.scatter(quantum_risks, quantum_returns, c='red', alpha=0.7, label='Quantum')
    
    plt.xlabel('Portfolio Risk')
    plt.ylabel('Expected Return')
    plt.title('Risk-Return Scatter Plot')
    plt.legend()
    plt.grid(True)
    
    # Portfolio weights comparison
    plt.subplot(1, 3, 3)
    if classical_frontier and quantum_frontier:
        # Compare minimum risk portfolios
        classical_min_risk = min(classical_frontier, key=lambda x: x['risk'])
        quantum_min_risk = min(quantum_frontier, key=lambda x: x['risk'])
        
        x = np.arange(len(classical_min_risk['weights']))
        width = 0.35
        
        plt.bar(x - width/2, classical_min_risk['weights'], width, label='Classical', alpha=0.7)
        plt.bar(x + width/2, quantum_min_risk['weights'], width, label='Quantum', alpha=0.7)
        
        plt.xlabel('Asset')
        plt.ylabel('Weight')
        plt.title('Minimum Risk Portfolio Weights')
        plt.legend()
        plt.xticks(x, [f'Asset_{i}' for i in range(len(classical_min_risk['weights']))])
    
    plt.tight_layout()
    plt.show()
    
    return classical_frontier, quantum_frontier

def quantum_portfolio_analysis():
    """
    Analyze quantum portfolio optimization properties
    """
    print("=== Quantum Portfolio Optimization Analysis ===\n")
    
    # Generate data
    classical_optimizer = ClassicalPortfolioOptimizer()
    returns_data = classical_optimizer.generate_credit_assets(n_assets=6, n_periods=252)
    expected_returns, cov_matrix, risk_free_rate = classical_optimizer.calculate_statistics(returns_data)
    
    # Create quantum optimizer
    quantum_optimizer = QuantumPortfolioOptimizer(num_qubits=4)
    
    # Analyze different target returns
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 8)
    optimization_results = []
    
    for target_return in target_returns:
        print(f"Optimizing for target return: {target_return:.4f}")
        
        # Classical optimization
        classical_success = classical_optimizer.markowitz_optimization(
            expected_returns, cov_matrix, target_return
        )
        
        # Quantum optimization
        quantum_success = quantum_optimizer.optimize_portfolio(
            expected_returns[:4], cov_matrix.iloc[:4, :4], target_return
        )
        
        if classical_success and quantum_success:
            optimization_results.append({
                'target_return': target_return,
                'classical_return': classical_optimizer.expected_return,
                'classical_risk': classical_optimizer.risk,
                'quantum_return': quantum_optimizer.expected_return,
                'quantum_risk': quantum_optimizer.risk
            })
    
    # Analyze results
    if optimization_results:
        results_df = pd.DataFrame(optimization_results)
        
        print("\nOptimization Results Summary:")
        print(f"   Classical Return Error: {np.mean(np.abs(results_df['target_return'] - results_df['classical_return'])):.4f}")
        print(f"   Quantum Return Error: {np.mean(np.abs(results_df['target_return'] - results_df['quantum_return'])):.4f}")
        print(f"   Classical Average Risk: {results_df['classical_risk'].mean():.4f}")
        print(f"   Quantum Average Risk: {results_df['quantum_risk'].mean():.4f}")
        
        # Plot analysis
        plt.figure(figsize=(12, 5))
        
        # Return accuracy
        plt.subplot(1, 2, 1)
        plt.plot(results_df['target_return'], results_df['classical_return'], 'b-o', label='Classical')
        plt.plot(results_df['target_return'], results_df['quantum_return'], 'r-s', label='Quantum')
        plt.plot(results_df['target_return'], results_df['target_return'], 'k--', label='Target')
        plt.xlabel('Target Return')
        plt.ylabel('Achieved Return')
        plt.title('Return Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Risk comparison
        plt.subplot(1, 2, 2)
        plt.plot(results_df['target_return'], results_df['classical_risk'], 'b-o', label='Classical')
        plt.plot(results_df['target_return'], results_df['quantum_risk'], 'r-s', label='Quantum')
        plt.xlabel('Target Return')
        plt.ylabel('Portfolio Risk')
        plt.title('Risk Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return optimization_results

# Run demos
if __name__ == "__main__":
    print("Running Portfolio Optimization Comparison...")
    classical_frontier, quantum_frontier = compare_portfolio_optimization()
    
    print("\nRunning Quantum Portfolio Analysis...")
    analysis_results = quantum_portfolio_analysis()
```

## 📊 Kết quả và Phân tích

### **Quantum Portfolio Optimization Advantages:**

#### **1. Quantum Properties:**
- **Superposition**: Parallel evaluation of multiple portfolios
- **Entanglement**: Complex asset correlations
- **Quantum Parallelism**: Exponential speedup potential

#### **2. Credit-specific Benefits:**
- **Non-linear Risk Models**: Quantum circuits capture complex risk relationships
- **High-dimensional Optimization**: Handle many assets efficiently
- **Quantum Advantage**: Potential speedup for large portfolios

#### **3. Performance Characteristics:**
- **Better Risk Modeling**: Quantum features improve risk estimation
- **Robustness**: Quantum optimization handles noisy market data
- **Scalability**: Quantum advantage for large-scale portfolio optimization

### **Comparison với Classical Portfolio Optimization:**

#### **Classical Limitations:**
- Limited to linear risk models
- Curse of dimensionality
- Local optima problems
- Assumption of normal returns

#### **Quantum Advantages:**
- Non-linear risk modeling
- High-dimensional optimization
- Global optimization potential
- Flexible return distributions

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Portfolio Calibration**
Implement quantum portfolio calibration methods cho credit risk management.

### **Exercise 2: Quantum Portfolio Ensemble Methods**
Build ensemble of quantum portfolio optimizers cho improved performance.

### **Exercise 3: Quantum Portfolio Feature Selection**
Develop quantum feature selection cho portfolio optimization.

### **Exercise 4: Quantum Portfolio Validation**
Create validation framework cho quantum portfolio models.

---

> *"Quantum portfolio optimization leverages quantum superposition and entanglement to provide superior risk-return optimization for credit portfolios."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Risk Measures (VaR, CVaR)](Day17.md) 