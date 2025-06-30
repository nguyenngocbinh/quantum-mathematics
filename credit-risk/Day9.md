# Ngày 9: Quantum Optimization cho Portfolio

## 🎯 Mục tiêu học tập

- Hiểu sâu về quantum optimization algorithms và classical optimization
- Nắm vững cách quantum optimization cải thiện portfolio management
- Implement quantum-enhanced portfolio optimization cho credit risk
- So sánh performance giữa quantum và classical optimization approaches

## 📚 Lý thuyết

### **Portfolio Optimization Fundamentals**

#### **1. Classical Portfolio Optimization**

**Markowitz Model:**
```
min w'Σw
s.t. w'μ = target_return
     w'1 = 1
     w ≥ 0
```

**Risk Measures:**
- **VaR**: Value at Risk
- **CVaR**: Conditional Value at Risk
- **Sharpe Ratio**: Risk-adjusted return

#### **2. Quantum Optimization Algorithms**

**Quantum Approximate Optimization Algorithm (QAOA):**
```
|ψ(β,γ)⟩ = ∏ᵢ e^(-iβᵢHₘ) ∏ᵢ e^(-iγᵢHₚ) |+⟩^⊗ⁿ
```

**Variational Quantum Eigensolver (VQE):**
```
min ⟨ψ(θ)|H|ψ(θ)⟩
```

**Quantum Adiabatic Algorithm:**
```
H(t) = (1-t/T)H₀ + (t/T)H₁
```

### **Quantum Portfolio Optimization**

#### **1. Portfolio State Representation:**

**Quantum Portfolio State:**
```
|portfolio⟩ = Σᵢ wᵢ|asset_i⟩
```

**Risk State:**
```
|risk⟩ = Σᵢ σᵢ|risk_factor_i⟩
```

#### **2. Quantum Cost Functions:**

**Portfolio Risk:**
```
H_risk = Σᵢⱼ wᵢwⱼσᵢⱼ|i⟩⟨j|
```

**Return Constraint:**
```
H_return = Σᵢ μᵢwᵢ|i⟩⟨i|
```

**Budget Constraint:**
```
H_budget = (Σᵢ wᵢ - 1)²
```

#### **3. Quantum Constraints:**

**Equality Constraints:**
```
⟨ψ|C|ψ⟩ = 0
```

**Inequality Constraints:**
```
⟨ψ|C|ψ⟩ ≥ 0
```

## 💻 Thực hành

### **Project 9: Quantum Portfolio Optimization System**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, Operator
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.circuit.library import TwoLocal, RealAmplitudes
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
import pennylane as qml

class ClassicalPortfolioOptimizer:
    """Classical portfolio optimization"""
    
    def __init__(self):
        pass
    
    def markowitz_optimization(self, returns, cov_matrix, target_return=None, risk_free_rate=0.02):
        """
        Classical Markowitz portfolio optimization
        """
        from scipy.optimize import minimize
        
        n_assets = len(returns)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return weights.T @ cov_matrix @ weights
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Budget constraint
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.sum(x * returns) - target_return
            })
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x, result.fun
    
    def calculate_portfolio_metrics(self, weights, returns, cov_matrix, risk_free_rate=0.02):
        """
        Calculate portfolio metrics
        """
        portfolio_return = np.sum(weights * returns)
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
        
        return {
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'weights': weights
        }

class QuantumPortfolioOptimizer:
    """Quantum portfolio optimization"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_portfolio_hamiltonian(self, returns, cov_matrix, target_return=None):
        """
        Create quantum Hamiltonian for portfolio optimization
        """
        n_assets = len(returns)
        
        # Risk Hamiltonian
        H_risk = np.zeros((2**self.num_qubits, 2**self.num_qubits))
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i < self.num_qubits and j < self.num_qubits:
                    # Encode covariance in Hamiltonian
                    H_risk[2**i, 2**j] = cov_matrix[i, j]
        
        # Return Hamiltonian
        H_return = np.zeros((2**self.num_qubits, 2**self.num_qubits))
        for i in range(n_assets):
            if i < self.num_qubits:
                H_return[2**i, 2**i] = returns[i]
        
        # Budget constraint Hamiltonian
        H_budget = np.zeros((2**self.num_qubits, 2**self.num_qubits))
        for i in range(n_assets):
            if i < self.num_qubits:
                H_budget[2**i, 2**i] = 1
        
        return H_risk, H_return, H_budget
    
    def create_vqe_circuit(self, returns, cov_matrix, target_return=None):
        """
        Create VQE circuit for portfolio optimization
        """
        # Create ansatz
        ansatz = RealAmplitudes(self.num_qubits, reps=2)
        
        # Create cost Hamiltonian
        H_risk, H_return, H_budget = self.create_portfolio_hamiltonian(returns, cov_matrix, target_return)
        
        # Combine Hamiltonians
        H_cost = H_risk + 0.1 * H_return + 0.5 * H_budget
        
        # Create VQE
        optimizer = SPSA(maxiter=100)
        vqe = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=self.backend
        )
        
        return vqe, H_cost
    
    def optimize_portfolio_vqe(self, returns, cov_matrix, target_return=None):
        """
        Optimize portfolio using VQE
        """
        vqe, H_cost = self.create_vqe_circuit(returns, cov_matrix, target_return)
        
        # Run VQE
        result = vqe.solve(H_cost)
        
        # Extract optimal parameters
        optimal_params = result.optimal_parameters
        
        # Create circuit with optimal parameters
        circuit = RealAmplitudes(self.num_qubits, reps=2)
        circuit.assign_parameters(optimal_params)
        
        # Measure circuit
        circuit.measure_all()
        
        # Execute circuit
        job = execute(circuit, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Convert to portfolio weights
        weights = self._extract_weights_from_counts(counts, len(returns))
        
        return weights, result.optimal_value
    
    def create_qaoa_circuit(self, returns, cov_matrix, target_return=None, p=2):
        """
        Create QAOA circuit for portfolio optimization
        """
        # Create cost Hamiltonian
        H_risk, H_return, H_budget = self.create_portfolio_hamiltonian(returns, cov_matrix, target_return)
        H_cost = H_risk + 0.1 * H_return + 0.5 * H_budget
        
        # Create QAOA
        optimizer = SPSA(maxiter=100)
        qaoa = QAOA(
            optimizer=optimizer,
            reps=p,
            quantum_instance=self.backend
        )
        
        return qaoa, H_cost
    
    def optimize_portfolio_qaoa(self, returns, cov_matrix, target_return=None, p=2):
        """
        Optimize portfolio using QAOA
        """
        qaoa, H_cost = self.create_qaoa_circuit(returns, cov_matrix, target_return, p)
        
        # Run QAOA
        result = qaoa.solve(H_cost)
        
        # Extract optimal parameters
        optimal_params = result.optimal_parameters
        
        # Create circuit with optimal parameters
        circuit = qaoa.construct_circuit(optimal_params, H_cost)
        
        # Measure circuit
        circuit.measure_all()
        
        # Execute circuit
        job = execute(circuit, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Convert to portfolio weights
        weights = self._extract_weights_from_counts(counts, len(returns))
        
        return weights, result.optimal_value
    
    def _extract_weights_from_counts(self, counts, n_assets):
        """
        Extract portfolio weights from measurement counts
        """
        total_shots = sum(counts.values())
        weights = np.zeros(n_assets)
        
        for state, count in counts.items():
            # Convert binary state to weights
            for i in range(min(len(state), n_assets)):
                if state[i] == '1':
                    weights[i] += count / total_shots
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        return weights

class QuantumCreditPortfolioOptimizer:
    """Quantum portfolio optimization with credit risk constraints"""
    
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.qpo = QuantumPortfolioOptimizer(num_qubits)
        
    def create_credit_risk_hamiltonian(self, returns, cov_matrix, default_probs, exposures):
        """
        Create Hamiltonian including credit risk
        """
        n_assets = len(returns)
        
        # Standard portfolio Hamiltonians
        H_risk, H_return, H_budget = self.qpo.create_portfolio_hamiltonian(returns, cov_matrix)
        
        # Credit risk Hamiltonian
        H_credit = np.zeros((2**self.num_qubits, 2**self.num_qubits))
        
        for i in range(n_assets):
            if i < self.num_qubits:
                # Credit risk contribution
                credit_risk = default_probs[i] * exposures[i]
                H_credit[2**i, 2**i] = credit_risk
        
        # Combine all Hamiltonians
        H_total = H_risk + 0.1 * H_return + 0.5 * H_budget + 0.3 * H_credit
        
        return H_total
    
    def optimize_credit_portfolio(self, returns, cov_matrix, default_probs, exposures, target_return=None):
        """
        Optimize portfolio with credit risk constraints
        """
        # Create credit risk Hamiltonian
        H_total = self.create_credit_risk_hamiltonian(returns, cov_matrix, default_probs, exposures)
        
        # Use VQE for optimization
        vqe, _ = self.qpo.create_vqe_circuit(returns, cov_matrix, target_return)
        
        # Run optimization
        result = vqe.solve(H_total)
        
        # Extract weights
        optimal_params = result.optimal_parameters
        circuit = RealAmplitudes(self.num_qubits, reps=2)
        circuit.assign_parameters(optimal_params)
        circuit.measure_all()
        
        # Execute circuit
        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Extract weights
        weights = self.qpo._extract_weights_from_counts(counts, len(returns))
        
        return weights, result.optimal_value

def compare_portfolio_optimization():
    """
    Compare classical and quantum portfolio optimization
    """
    print("=== Classical vs Quantum Portfolio Optimization ===\n")
    
    # Generate sample data
    np.random.seed(42)
    n_assets = 4
    
    # Asset returns and covariance
    returns = np.array([0.08, 0.12, 0.06, 0.10])
    cov_matrix = np.array([
        [0.04, 0.02, 0.01, 0.03],
        [0.02, 0.09, 0.02, 0.04],
        [0.01, 0.02, 0.06, 0.02],
        [0.03, 0.04, 0.02, 0.08]
    ])
    
    # Credit risk data
    default_probs = np.array([0.02, 0.03, 0.01, 0.025])
    exposures = np.array([1000000, 1500000, 800000, 1200000])
    
    # Classical optimization
    print("1. Classical Markowitz Optimization:")
    cpo = ClassicalPortfolioOptimizer()
    
    classical_weights, classical_risk = cpo.markowitz_optimization(returns, cov_matrix)
    classical_metrics = cpo.calculate_portfolio_metrics(classical_weights, returns, cov_matrix)
    
    print(f"   Weights: {classical_weights}")
    print(f"   Return: {classical_metrics['return']:.4f}")
    print(f"   Risk: {classical_metrics['risk']:.4f}")
    print(f"   Sharpe Ratio: {classical_metrics['sharpe_ratio']:.4f}")
    
    # Quantum VQE optimization
    print("\n2. Quantum VQE Optimization:")
    qpo = QuantumPortfolioOptimizer(num_qubits=4)
    
    quantum_weights_vqe, quantum_cost_vqe = qpo.optimize_portfolio_vqe(returns, cov_matrix)
    quantum_metrics_vqe = cpo.calculate_portfolio_metrics(quantum_weights_vqe, returns, cov_matrix)
    
    print(f"   Weights: {quantum_weights_vqe}")
    print(f"   Return: {quantum_metrics_vqe['return']:.4f}")
    print(f"   Risk: {quantum_metrics_vqe['risk']:.4f}")
    print(f"   Sharpe Ratio: {quantum_metrics_vqe['sharpe_ratio']:.4f}")
    
    # Quantum QAOA optimization
    print("\n3. Quantum QAOA Optimization:")
    quantum_weights_qaoa, quantum_cost_qaoa = qpo.optimize_portfolio_qaoa(returns, cov_matrix)
    quantum_metrics_qaoa = cpo.calculate_portfolio_metrics(quantum_weights_qaoa, returns, cov_matrix)
    
    print(f"   Weights: {quantum_weights_qaoa}")
    print(f"   Return: {quantum_metrics_qaoa['return']:.4f}")
    print(f"   Risk: {quantum_metrics_qaoa['risk']:.4f}")
    print(f"   Sharpe Ratio: {quantum_metrics_qaoa['sharpe_ratio']:.4f}")
    
    # Credit risk optimization
    print("\n4. Quantum Credit Risk Optimization:")
    qcpo = QuantumCreditPortfolioOptimizer(num_qubits=4)
    
    credit_weights, credit_cost = qcpo.optimize_credit_portfolio(returns, cov_matrix, default_probs, exposures)
    credit_metrics = cpo.calculate_portfolio_metrics(credit_weights, returns, cov_matrix)
    
    print(f"   Weights: {credit_weights}")
    print(f"   Return: {credit_metrics['return']:.4f}")
    print(f"   Risk: {credit_metrics['risk']:.4f}")
    print(f"   Sharpe Ratio: {credit_metrics['sharpe_ratio']:.4f}")
    
    # Compare results
    print(f"\n5. Comparison:")
    methods = ['Classical', 'VQE', 'QAOA', 'Credit']
    sharpe_ratios = [
        classical_metrics['sharpe_ratio'],
        quantum_metrics_vqe['sharpe_ratio'],
        quantum_metrics_qaoa['sharpe_ratio'],
        credit_metrics['sharpe_ratio']
    ]
    
    for method, sharpe in zip(methods, sharpe_ratios):
        print(f"   {method} Sharpe Ratio: {sharpe:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Weights comparison
    plt.subplot(1, 3, 1)
    x = np.arange(n_assets)
    width = 0.2
    
    plt.bar(x - 1.5*width, classical_weights, width, label='Classical', color='blue')
    plt.bar(x - 0.5*width, quantum_weights_vqe, width, label='VQE', color='orange')
    plt.bar(x + 0.5*width, quantum_weights_qaoa, width, label='QAOA', color='green')
    plt.bar(x + 1.5*width, credit_weights, width, label='Credit', color='red')
    
    plt.xlabel('Asset')
    plt.ylabel('Weight')
    plt.title('Portfolio Weights Comparison')
    plt.xticks(x, [f'Asset {i+1}' for i in range(n_assets)])
    plt.legend()
    
    # Risk-return comparison
    plt.subplot(1, 3, 2)
    risks = [
        classical_metrics['risk'],
        quantum_metrics_vqe['risk'],
        quantum_metrics_qaoa['risk'],
        credit_metrics['risk']
    ]
    returns_plot = [
        classical_metrics['return'],
        quantum_metrics_vqe['return'],
        quantum_metrics_qaoa['return'],
        credit_metrics['return']
    ]
    
    plt.scatter(risks, returns_plot, c=['blue', 'orange', 'green', 'red'], s=100)
    for i, method in enumerate(methods):
        plt.annotate(method, (risks[i], returns_plot[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.title('Risk-Return Comparison')
    plt.grid(True)
    
    # Sharpe ratio comparison
    plt.subplot(1, 3, 3)
    plt.bar(methods, sharpe_ratios, color=['blue', 'orange', 'green', 'red'])
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return classical_weights, quantum_weights_vqe, quantum_weights_qaoa, credit_weights

def quantum_optimization_convergence():
    """
    Analyze convergence of quantum optimization algorithms
    """
    print("=== Quantum Optimization Convergence Analysis ===\n")
    
    # Generate test data
    returns = np.array([0.08, 0.12, 0.06, 0.10])
    cov_matrix = np.array([
        [0.04, 0.02, 0.01, 0.03],
        [0.02, 0.09, 0.02, 0.04],
        [0.01, 0.02, 0.06, 0.02],
        [0.03, 0.04, 0.02, 0.08]
    ])
    
    # Test different optimization parameters
    max_iters = [50, 100, 200, 500]
    qpo = QuantumPortfolioOptimizer(num_qubits=4)
    
    vqe_costs = []
    qaoa_costs = []
    
    for max_iter in max_iters:
        print(f"Testing with {max_iter} iterations...")
        
        # VQE with different iterations
        optimizer_vqe = SPSA(maxiter=max_iter)
        vqe = VQE(
            ansatz=RealAmplitudes(4, reps=2),
            optimizer=optimizer_vqe,
            quantum_instance=Aer.get_backend('qasm_simulator')
        )
        
        H_risk, H_return, H_budget = qpo.create_portfolio_hamiltonian(returns, cov_matrix)
        H_cost = H_risk + 0.1 * H_return + 0.5 * H_budget
        
        result_vqe = vqe.solve(H_cost)
        vqe_costs.append(result_vqe.optimal_value)
        
        # QAOA with different iterations
        optimizer_qaoa = SPSA(maxiter=max_iter)
        qaoa = QAOA(
            optimizer=optimizer_qaoa,
            reps=2,
            quantum_instance=Aer.get_backend('qasm_simulator')
        )
        
        result_qaoa = qaoa.solve(H_cost)
        qaoa_costs.append(result_qaoa.optimal_value)
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(max_iters, vqe_costs, 'o-', label='VQE', linewidth=2, markersize=8)
    plt.xlabel('Max Iterations')
    plt.ylabel('Optimal Cost')
    plt.title('VQE Convergence')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(max_iters, qaoa_costs, 's-', label='QAOA', linewidth=2, markersize=8)
    plt.xlabel('Max Iterations')
    plt.ylabel('Optimal Cost')
    plt.title('QAOA Convergence')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return max_iters, vqe_costs, qaoa_costs

# Exercise: Quantum Portfolio Rebalancing
def quantum_portfolio_rebalancing():
    """
    Exercise: Implement quantum portfolio rebalancing
    """
    # Initial portfolio
    initial_weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Market changes
    returns = np.array([0.08, 0.12, 0.06, 0.10])
    cov_matrix = np.array([
        [0.04, 0.02, 0.01, 0.03],
        [0.02, 0.09, 0.02, 0.04],
        [0.01, 0.02, 0.06, 0.02],
        [0.03, 0.04, 0.02, 0.08]
    ])
    
    # Transaction costs
    transaction_costs = 0.001  # 0.1% per trade
    
    def rebalancing_objective(new_weights):
        """
        Objective function for rebalancing
        """
        # Portfolio risk
        risk = new_weights.T @ cov_matrix @ new_weights
        
        # Transaction costs
        trades = np.abs(new_weights - initial_weights)
        transaction_cost = np.sum(trades) * transaction_costs
        
        return risk + transaction_cost
    
    # Classical rebalancing
    from scipy.optimize import minimize
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]
    bounds = [(0, 1) for _ in range(4)]
    
    result = minimize(
        rebalancing_objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    classical_rebalanced = result.x
    
    # Quantum rebalancing
    qpo = QuantumPortfolioOptimizer(num_qubits=4)
    
    # Create Hamiltonian including transaction costs
    H_risk, H_return, H_budget = qpo.create_portfolio_hamiltonian(returns, cov_matrix)
    
    # Add transaction cost term
    H_transaction = np.zeros((16, 16))
    for i in range(4):
        H_transaction[2**i, 2**i] = transaction_costs
    
    H_total = H_risk + 0.1 * H_return + 0.5 * H_budget + 0.2 * H_transaction
    
    quantum_weights, quantum_cost = qpo.optimize_portfolio_vqe(returns, cov_matrix)
    
    print("=== Portfolio Rebalancing ===")
    print(f"Initial Weights: {initial_weights}")
    print(f"Classical Rebalanced: {classical_rebalanced}")
    print(f"Quantum Rebalanced: {quantum_weights}")
    
    # Calculate rebalancing metrics
    classical_trades = np.sum(np.abs(classical_rebalanced - initial_weights))
    quantum_trades = np.sum(np.abs(quantum_weights - initial_weights))
    
    print(f"Classical Total Trades: {classical_trades:.4f}")
    print(f"Quantum Total Trades: {quantum_trades:.4f}")
    
    return initial_weights, classical_rebalanced, quantum_weights

# Run demos
if __name__ == "__main__":
    print("Running Portfolio Optimization Comparisons...")
    classical_weights, vqe_weights, qaoa_weights, credit_weights = compare_portfolio_optimization()
    
    print("\nRunning Convergence Analysis...")
    max_iters, vqe_costs, qaoa_costs = quantum_optimization_convergence()
    
    print("\nRunning Portfolio Rebalancing Exercise...")
    initial_weights, classical_rebalanced, quantum_rebalanced = quantum_portfolio_rebalancing()
```

### **Exercise 2: Quantum Optimization for Risk Management**

```python
def quantum_risk_management():
    """
    Exercise: Implement quantum optimization for risk management
    """
    # Risk management parameters
    risk_budget = 0.1  # 10% risk budget
    n_assets = 4
    
    # Asset risk contributions
    asset_risks = np.array([0.05, 0.08, 0.03, 0.06])
    
    def risk_budget_objective(weights):
        """
        Objective function for risk budget optimization
        """
        # Risk contribution
        risk_contrib = np.sum(weights * asset_risks)
        
        # Penalty for exceeding risk budget
        penalty = max(0, risk_contrib - risk_budget) * 100
        
        return risk_contrib + penalty
    
    # Classical optimization
    from scipy.optimize import minimize
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]
    bounds = [(0, 1) for _ in range(n_assets)]
    
    result = minimize(
        risk_budget_objective,
        np.ones(n_assets) / n_assets,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    classical_risk_weights = result.x
    
    # Quantum optimization
    qpo = QuantumPortfolioOptimizer(num_qubits=4)
    
    # Create risk budget Hamiltonian
    H_risk_budget = np.zeros((16, 16))
    for i in range(n_assets):
        H_risk_budget[2**i, 2**i] = asset_risks[i]
    
    # Add penalty for exceeding budget
    H_penalty = np.zeros((16, 16))
    for i in range(16):
        # Calculate risk contribution for this state
        risk_contrib = 0
        for j in range(n_assets):
            if (i >> j) & 1:
                risk_contrib += asset_risks[j]
        
        if risk_contrib > risk_budget:
            H_penalty[i, i] = 100
    
    H_total = H_risk_budget + H_penalty
    
    quantum_risk_weights, quantum_cost = qpo.optimize_portfolio_vqe(
        np.ones(n_assets), np.eye(n_assets)
    )
    
    print("=== Risk Budget Optimization ===")
    print(f"Risk Budget: {risk_budget}")
    print(f"Asset Risks: {asset_risks}")
    print(f"Classical Weights: {classical_risk_weights}")
    print(f"Quantum Weights: {quantum_risk_weights}")
    
    # Calculate risk contributions
    classical_risk = np.sum(classical_risk_weights * asset_risks)
    quantum_risk = np.sum(quantum_risk_weights * asset_risks)
    
    print(f"Classical Risk Contribution: {classical_risk:.4f}")
    print(f"Quantum Risk Contribution: {quantum_risk:.4f}")
    
    return classical_risk_weights, quantum_risk_weights

# Run exercises
if __name__ == "__main__":
    print("Running Risk Management Exercise...")
    classical_risk_weights, quantum_risk_weights = quantum_risk_management()
```

## 📊 Kết quả và Phân tích

### **Quantum Optimization Advantages:**

#### **1. Speedup:**
- **QAOA**: Potential quantum advantage cho certain problems
- **VQE**: Hybrid quantum-classical optimization
- **Adiabatic**: Quantum annealing approaches

#### **2. Solution Quality:**
- **Global Optimization**: Better exploration of solution space
- **Quantum Tunneling**: Escape local minima
- **Quantum Interference**: Enhanced optimization

#### **3. Financial Applications:**
- **Portfolio Optimization**: Quantum asset allocation
- **Risk Management**: Quantum risk budgeting
- **Credit Risk**: Quantum credit portfolio optimization

### **Comparison với Classical Optimization:**

#### **Classical Limitations:**
- Local optimization methods
- Limited by computational resources
- Sequential processing

#### **Quantum Advantages:**
- Global optimization capabilities
- Quantum speedup cho certain problems
- Parallel processing of solutions

## 🎯 Bài tập về nhà

### **Exercise 1: Quantum Optimization Calibration**
Implement quantum optimization calibration cho portfolio models.

### **Exercise 2: Quantum Optimization Networks**
Build quantum optimization cho network-based portfolio models.

### **Exercise 3: Quantum Optimization Benchmarking**
Develop benchmarking framework cho quantum optimization algorithms.

### **Exercise 4: Quantum Optimization Validation**
Create validation framework cho quantum optimization results.

---

> *"Quantum optimization algorithms provide new capabilities for portfolio management, enabling more sophisticated risk-return optimization."* - Quantum Finance Research

> Ngày tiếp theo: [Quantum Machine Learning Basics cho Finance](Day10.md) 