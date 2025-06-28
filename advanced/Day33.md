# Day 33: Quantum Simulation Projects

## 🎯 Mục tiêu
- Triển khai các dự án simulation lượng tử thực tế
- Mô phỏng hệ thống hóa học và vật liệu
- Giải quyết bài toán tối ưu hóa phức tạp
- Xử lý dữ liệu thực tế với quantum algorithms

## 🧪 Quantum Simulation Projects - Tổng Quan

### Tại sao Quantum Simulation?
- **Quantum advantage**: Mô phỏng hệ thống lượng tử hiệu quả hơn
- **Chemistry simulation**: Tính toán cấu trúc phân tử và phản ứng
- **Material science**: Thiết kế vật liệu mới với tính chất đặc biệt
- **Optimization**: Giải quyết bài toán tối ưu hóa phức tạp
- **Real-world applications**: Áp dụng vào các vấn đề thực tế

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms import VQE, QAOA
from qiskit.algorithms.optimizers import SPSA, COBYLA, ADAM
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow import I, X, Y, Z
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Optional
import seaborn as sns
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

## 🔬 Project 1: Quantum Chemistry Simulation

### 1. Molecular Energy Calculation

```python
class QuantumChemistrySimulator:
    """
    Simulator hóa học lượng tử sử dụng VQE
    """
    
    def __init__(self, backend='qasm_simulator'):
        self.backend = Aer.get_backend(backend)
        self.optimizer = SPSA(maxiter=100)
        
    def create_molecular_hamiltonian(self, molecule: str) -> SparsePauliOp:
        """
        Tạo Hamiltonian cho phân tử
        """
        try:
            # Sử dụng PySCF driver
            driver = PySCFDriver(atom=molecule)
            problem = driver.run()
            
            # Map sang qubits
            mapper = JordanWignerMapper()
            qubit_op = mapper.map(problem.second_q_ops()[0])
            
            return qubit_op
        except Exception as e:
            print(f"Error creating Hamiltonian: {e}")
            # Fallback: tạo Hamiltonian đơn giản
            return self._create_simple_hamiltonian()
    
    def _create_simple_hamiltonian(self) -> SparsePauliOp:
        """
        Tạo Hamiltonian đơn giản cho demo
        """
        # H2 molecule Hamiltonian (simplified)
        coeffs = [0.011280, 0.171201, 0.171201, 0.171201, 0.171201, -0.222796, -0.222796, 0.174348]
        paulis = ['II', 'IZ', 'ZI', 'ZZ', 'XX', 'YY', 'YY', 'ZZ']
        
        hamiltonian = SparsePauliOp.from_list(list(zip(paulis, coeffs)))
        return hamiltonian
    
    def create_ansatz(self, n_qubits: int, depth: int = 2) -> QuantumCircuit:
        """
        Tạo ansatz circuit cho VQE
        """
        ansatz = TwoLocal(n_qubits, ['ry', 'rz'], 'cz', reps=depth, entanglement='linear')
        return ansatz
    
    def run_vqe_simulation(self, molecule: str, n_qubits: int = 2) -> Dict:
        """
        Chạy VQE simulation cho phân tử
        """
        # Tạo Hamiltonian
        hamiltonian = self.create_molecular_hamiltonian(molecule)
        
        # Tạo ansatz
        ansatz = self.create_ansatz(n_qubits)
        
        # Khởi tạo VQE
        vqe = VQE(ansatz, optimizer=self.optimizer, quantum_instance=self.backend)
        
        # Chạy VQE
        result = vqe.solve(hamiltonian)
        
        return {
            'ground_state_energy': result.eigenvalue.real,
            'optimal_parameters': result.optimal_parameters,
            'optimization_history': result.optimizer_history,
            'hamiltonian': hamiltonian,
            'ansatz': ansatz
        }

# Test quantum chemistry simulation
print("Quantum Chemistry Simulation Demo:")
chem_sim = QuantumChemistrySimulator()

# Simulate H2 molecule
h2_result = chem_sim.run_vqe_simulation("H .0 .0 .0; H .0 .0 0.74", n_qubits=2)

print(f"H2 Ground State Energy: {h2_result['ground_state_energy']:.6f} Hartree")
print(f"Optimal Parameters: {h2_result['optimal_parameters']}")
```

### 2. Potential Energy Surface

```python
def calculate_potential_energy_surface(molecule_base: str, bond_lengths: List[float]) -> Dict:
    """
    Tính toán potential energy surface
    """
    energies = []
    chem_sim = QuantumChemistrySimulator()
    
    for bond_length in bond_lengths:
        # Tạo phân tử với bond length khác nhau
        if "H" in molecule_base:
            molecule = f"H .0 .0 .0; H .0 .0 {bond_length}"
        else:
            molecule = molecule_base  # Placeholder for other molecules
        
        try:
            result = chem_sim.run_vqe_simulation(molecule)
            energies.append(result['ground_state_energy'])
        except:
            energies.append(np.nan)
    
    return {
        'bond_lengths': bond_lengths,
        'energies': energies,
        'molecule': molecule_base
    }

def plot_potential_energy_surface(data: Dict):
    """
    Vẽ potential energy surface
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data['bond_lengths'], data['energies'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Bond Length (Å)')
    plt.ylabel('Energy (Hartree)')
    plt.title(f'Potential Energy Surface for {data["molecule"]}')
    plt.grid(True, alpha=0.3)
    plt.show()

# Calculate and plot potential energy surface
print("\nPotential Energy Surface Calculation:")
bond_lengths = np.linspace(0.5, 2.0, 20)
pes_data = calculate_potential_energy_surface("H2", bond_lengths)

print("Bond Lengths vs Energies:")
for i, (length, energy) in enumerate(zip(pes_data['bond_lengths'], pes_data['energies'])):
    if not np.isnan(energy):
        print(f"  {length:.3f} Å: {energy:.6f} Hartree")

# Plot the surface
plot_potential_energy_surface(pes_data)
```

## 🏗️ Project 2: Material Science Simulation

### 1. Crystal Structure Optimization

```python
class QuantumMaterialSimulator:
    """
    Simulator vật liệu lượng tử
    """
    
    def __init__(self, backend='qasm_simulator'):
        self.backend = Aer.get_backend(backend)
        self.optimizer = COBYLA(maxiter=200)
    
    def create_lattice_hamiltonian(self, lattice_type: str, n_sites: int) -> SparsePauliOp:
        """
        Tạo Hamiltonian cho crystal lattice
        """
        if lattice_type == "1D_chain":
            return self._create_1d_chain_hamiltonian(n_sites)
        elif lattice_type == "2D_square":
            return self._create_2d_square_hamiltonian(n_sites)
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")
    
    def _create_1d_chain_hamiltonian(self, n_sites: int) -> SparsePauliOp:
        """
        Tạo Hamiltonian cho 1D chain
        """
        # Heisenberg model: H = J∑ᵢⱼ Sᵢ·Sⱼ
        J = 1.0  # Coupling strength
        
        hamiltonian_terms = []
        
        for i in range(n_sites - 1):
            # Nearest neighbor interactions
            # XX term
            xx_term = ['I'] * n_sites
            xx_term[i] = 'X'
            xx_term[i+1] = 'X'
            hamiltonian_terms.append((''.join(xx_term), J/4))
            
            # YY term
            yy_term = ['I'] * n_sites
            yy_term[i] = 'Y'
            yy_term[i+1] = 'Y'
            hamiltonian_terms.append((''.join(yy_term), J/4))
            
            # ZZ term
            zz_term = ['I'] * n_sites
            zz_term[i] = 'Z'
            zz_term[i+1] = 'Z'
            hamiltonian_terms.append((''.join(zz_term), J/4))
        
        return SparsePauliOp.from_list(hamiltonian_terms)
    
    def _create_2d_square_hamiltonian(self, n_sites: int) -> SparsePauliOp:
        """
        Tạo Hamiltonian cho 2D square lattice
        """
        # Simplified 2D model (4 sites)
        if n_sites != 4:
            raise ValueError("2D simulation currently supports only 4 sites")
        
        J = 1.0
        hamiltonian_terms = []
        
        # Horizontal bonds
        bonds = [(0,1), (2,3), (0,2), (1,3)]
        
        for i, j in bonds:
            # XX term
            xx_term = ['I'] * n_sites
            xx_term[i] = 'X'
            xx_term[j] = 'X'
            hamiltonian_terms.append((''.join(xx_term), J/4))
            
            # YY term
            yy_term = ['I'] * n_sites
            yy_term[i] = 'Y'
            yy_term[j] = 'Y'
            hamiltonian_terms.append((''.join(yy_term), J/4))
            
            # ZZ term
            zz_term = ['I'] * n_sites
            zz_term[i] = 'Z'
            zz_term[j] = 'Z'
            hamiltonian_terms.append((''.join(zz_term), J/4))
        
        return SparsePauliOp.from_list(hamiltonian_terms)
    
    def optimize_crystal_structure(self, lattice_type: str, n_sites: int) -> Dict:
        """
        Tối ưu hóa cấu trúc tinh thể
        """
        # Tạo Hamiltonian
        hamiltonian = self.create_lattice_hamiltonian(lattice_type, n_sites)
        
        # Tạo ansatz
        ansatz = EfficientSU2(n_sites, reps=2)
        
        # Chạy VQE
        vqe = VQE(ansatz, optimizer=self.optimizer, quantum_instance=self.backend)
        result = vqe.solve(hamiltonian)
        
        return {
            'ground_state_energy': result.eigenvalue.real,
            'optimal_parameters': result.optimal_parameters,
            'lattice_type': lattice_type,
            'n_sites': n_sites,
            'hamiltonian': hamiltonian
        }

# Test material science simulation
print("\nMaterial Science Simulation Demo:")
material_sim = QuantumMaterialSimulator()

# 1D chain simulation
chain_result = material_sim.optimize_crystal_structure("1D_chain", n_sites=4)
print(f"1D Chain Ground State Energy: {chain_result['ground_state_energy']:.6f}")

# 2D square lattice simulation
square_result = material_sim.optimize_crystal_structure("2D_square", n_sites=4)
print(f"2D Square Lattice Ground State Energy: {square_result['ground_state_energy']:.6f}")
```

### 2. Phase Transition Detection

```python
def detect_phase_transitions(material_sim: QuantumMaterialSimulator, 
                           coupling_range: List[float]) -> Dict:
    """
    Phát hiện phase transitions
    """
    energies = []
    order_parameters = []
    
    for J in coupling_range:
        # Tạo Hamiltonian với coupling strength J
        hamiltonian_terms = []
        n_sites = 4
        
        bonds = [(0,1), (2,3), (0,2), (1,3)]
        for i, j in bonds:
            # XX term
            xx_term = ['I'] * n_sites
            xx_term[i] = 'X'
            xx_term[j] = 'X'
            hamiltonian_terms.append((''.join(xx_term), J/4))
            
            # YY term
            yy_term = ['I'] * n_sites
            yy_term[i] = 'Y'
            yy_term[j] = 'Y'
            hamiltonian_terms.append((''.join(yy_term), J/4))
            
            # ZZ term
            zz_term = ['I'] * n_sites
            zz_term[i] = 'Z'
            zz_term[j] = 'Z'
            hamiltonian_terms.append((''.join(zz_term), J/4))
        
        hamiltonian = SparsePauliOp.from_list(hamiltonian_terms)
        
        # Tính ground state energy
        ansatz = EfficientSU2(n_sites, reps=2)
        vqe = VQE(ansatz, optimizer=material_sim.optimizer, 
                 quantum_instance=material_sim.backend)
        result = vqe.solve(hamiltonian)
        
        energies.append(result.eigenvalue.real)
        
        # Tính order parameter (simplified)
        order_param = abs(J)  # Simplified order parameter
        order_parameters.append(order_param)
    
    return {
        'coupling_strengths': coupling_range,
        'energies': energies,
        'order_parameters': order_parameters
    }

def plot_phase_transitions(data: Dict):
    """
    Vẽ phase transition
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Energy vs coupling strength
    ax1.plot(data['coupling_strengths'], data['energies'], 'bo-', linewidth=2)
    ax1.set_xlabel('Coupling Strength J')
    ax1.set_ylabel('Ground State Energy')
    ax1.set_title('Energy vs Coupling Strength')
    ax1.grid(True, alpha=0.3)
    
    # Order parameter vs coupling strength
    ax2.plot(data['coupling_strengths'], data['order_parameters'], 'ro-', linewidth=2)
    ax2.set_xlabel('Coupling Strength J')
    ax2.set_ylabel('Order Parameter')
    ax2.set_title('Order Parameter vs Coupling Strength')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Detect phase transitions
print("\nPhase Transition Detection:")
coupling_range = np.linspace(-2.0, 2.0, 20)
phase_data = detect_phase_transitions(material_sim, coupling_range)

print("Coupling Strength vs Energy:")
for i, (J, energy) in enumerate(zip(phase_data['coupling_strengths'], phase_data['energies'])):
    print(f"  J = {J:.2f}: E = {energy:.6f}")

# Plot phase transitions
plot_phase_transitions(phase_data)
```

## 🎯 Project 3: Optimization Problems

### 1. Traveling Salesman Problem (TSP)

```python
class QuantumTSPSolver:
    """
    Giải bài toán TSP sử dụng QAOA
    """
    
    def __init__(self, backend='qasm_simulator'):
        self.backend = Aer.get_backend(backend)
        self.optimizer = SPSA(maxiter=100)
    
    def create_tsp_hamiltonian(self, distances: List[List[float]]) -> SparsePauliOp:
        """
        Tạo Hamiltonian cho TSP
        """
        n_cities = len(distances)
        
        # Simplified TSP Hamiltonian
        # H = ∑ᵢⱼ dᵢⱼ xᵢ xⱼ + λ(∑ᵢ xᵢ - 1)²
        hamiltonian_terms = []
        
        # Distance terms
        for i in range(n_cities):
            for j in range(i+1, n_cities):
                if distances[i][j] > 0:
                    # Create term for cities i and j
                    term = ['I'] * n_cities
                    term[i] = 'Z'
                    term[j] = 'Z'
                    hamiltonian_terms.append((''.join(term), distances[i][j]/4))
        
        # Constraint terms (simplified)
        lambda_constraint = 10.0
        for i in range(n_cities):
            term = ['I'] * n_cities
            term[i] = 'Z'
            hamiltonian_terms.append((''.join(term), lambda_constraint/2))
        
        return SparsePauliOp.from_list(hamiltonian_terms)
    
    def solve_tsp(self, distances: List[List[float]]) -> Dict:
        """
        Giải TSP sử dụng QAOA
        """
        # Tạo Hamiltonian
        hamiltonian = self.create_tsp_hamiltonian(distances)
        
        # Chạy QAOA
        qaoa = QAOA(optimizer=self.optimizer, quantum_instance=self.backend)
        result = qaoa.solve(hamiltonian)
        
        return {
            'optimal_cost': result.eigenvalue.real,
            'optimal_parameters': result.optimal_parameters,
            'solution': result.eigenstate,
            'n_cities': len(distances)
        }

# Test TSP solver
print("\nTraveling Salesman Problem Demo:")
tsp_solver = QuantumTSPSolver()

# Create sample TSP instance
distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

tsp_result = tsp_solver.solve_tsp(distances)
print(f"TSP Optimal Cost: {tsp_result['optimal_cost']:.6f}")
print(f"Number of Cities: {tsp_result['n_cities']}")
```

### 2. Portfolio Optimization

```python
class QuantumPortfolioOptimizer:
    """
    Tối ưu hóa portfolio sử dụng quantum algorithms
    """
    
    def __init__(self, backend='qasm_simulator'):
        self.backend = Aer.get_backend(backend)
        self.optimizer = COBYLA(maxiter=150)
    
    def create_portfolio_hamiltonian(self, returns: List[float], 
                                   risk_matrix: List[List[float]], 
                                   target_return: float) -> SparsePauliOp:
        """
        Tạo Hamiltonian cho portfolio optimization
        """
        n_assets = len(returns)
        
        # Portfolio Hamiltonian: H = λ₁(μ - μₜ)² + λ₂σ²
        hamiltonian_terms = []
        
        # Return constraint
        lambda_return = 1.0
        for i in range(n_assets):
            term = ['I'] * n_assets
            term[i] = 'Z'
            hamiltonian_terms.append((''.join(term), lambda_return * returns[i] / 2))
        
        # Risk terms
        lambda_risk = 0.5
        for i in range(n_assets):
            for j in range(n_assets):
                if risk_matrix[i][j] > 0:
                    term = ['I'] * n_assets
                    term[i] = 'Z'
                    term[j] = 'Z'
                    hamiltonian_terms.append((''.join(term), lambda_risk * risk_matrix[i][j] / 4))
        
        return SparsePauliOp.from_list(hamiltonian_terms)
    
    def optimize_portfolio(self, returns: List[float], 
                         risk_matrix: List[List[float]], 
                         target_return: float) -> Dict:
        """
        Tối ưu hóa portfolio
        """
        # Tạo Hamiltonian
        hamiltonian = self.create_portfolio_hamiltonian(returns, risk_matrix, target_return)
        
        # Tạo ansatz
        n_assets = len(returns)
        ansatz = EfficientSU2(n_assets, reps=2)
        
        # Chạy VQE
        vqe = VQE(ansatz, optimizer=self.optimizer, quantum_instance=self.backend)
        result = vqe.solve(hamiltonian)
        
        return {
            'optimal_value': result.eigenvalue.real,
            'optimal_parameters': result.optimal_parameters,
            'n_assets': n_assets,
            'target_return': target_return
        }

# Test portfolio optimization
print("\nPortfolio Optimization Demo:")
portfolio_opt = QuantumPortfolioOptimizer()

# Sample data
returns = [0.08, 0.12, 0.06, 0.15]  # Expected returns
risk_matrix = [
    [0.04, 0.02, 0.01, 0.03],
    [0.02, 0.09, 0.02, 0.04],
    [0.01, 0.02, 0.06, 0.01],
    [0.03, 0.04, 0.01, 0.12]
]  # Risk covariance matrix

target_return = 0.10

portfolio_result = portfolio_opt.optimize_portfolio(returns, risk_matrix, target_return)
print(f"Portfolio Optimal Value: {portfolio_result['optimal_value']:.6f}")
print(f"Number of Assets: {portfolio_result['n_assets']}")
print(f"Target Return: {portfolio_result['target_return']:.2%}")
```

## 📊 Real-world Data Processing

### 1. Financial Data Analysis

```python
def quantum_financial_analysis():
    """
    Phân tích dữ liệu tài chính sử dụng quantum algorithms
    """
    # Generate synthetic financial data
    np.random.seed(42)
    n_days = 100
    n_stocks = 4
    
    # Generate price data
    prices = np.random.randn(n_days, n_stocks) * 0.02 + 1.0
    prices = np.cumprod(prices, axis=0)
    
    # Calculate returns
    returns = np.diff(prices, axis=0) / prices[:-1]
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(returns.T)
    
    # Calculate volatility
    volatility = np.std(returns, axis=0)
    
    print("Financial Data Analysis:")
    print(f"Number of days: {n_days}")
    print(f"Number of stocks: {n_stocks}")
    print(f"Average volatility: {np.mean(volatility):.4f}")
    
    # Create correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Stock Correlation Matrix')
    plt.show()
    
    return {
        'returns': returns,
        'correlation_matrix': correlation_matrix,
        'volatility': volatility,
        'prices': prices
    }

# Run financial analysis
financial_data = quantum_financial_analysis()
```

### 2. Climate Data Processing

```python
def quantum_climate_analysis():
    """
    Phân tích dữ liệu khí hậu sử dụng quantum algorithms
    """
    # Generate synthetic climate data
    np.random.seed(42)
    n_years = 50
    n_months = 12
    
    # Temperature data with trend and seasonality
    base_temp = 15.0
    trend = 0.02  # Warming trend
    seasonal_amplitude = 10.0
    
    temperatures = []
    for year in range(n_years):
        for month in range(n_months):
            # Add trend, seasonality, and noise
            temp = (base_temp + trend * year + 
                   seasonal_amplitude * np.sin(2 * np.pi * month / 12) +
                   np.random.normal(0, 2))
            temperatures.append(temp)
    
    temperatures = np.array(temperatures)
    
    # Analyze trends
    years = np.arange(n_years * n_months) / 12
    
    # Fit linear trend
    coeffs = np.polyfit(years, temperatures, 1)
    trend_line = np.polyval(coeffs, years)
    
    print("Climate Data Analysis:")
    print(f"Number of years: {n_years}")
    print(f"Temperature trend: {coeffs[0]:.4f}°C/year")
    print(f"Average temperature: {np.mean(temperatures):.2f}°C")
    
    # Plot temperature data
    plt.figure(figsize=(12, 6))
    plt.plot(years, temperatures, 'b-', alpha=0.7, label='Temperature')
    plt.plot(years, trend_line, 'r-', linewidth=2, label='Trend')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.title('Climate Temperature Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'temperatures': temperatures,
        'years': years,
        'trend': coeffs[0],
        'trend_line': trend_line
    }

# Run climate analysis
climate_data = quantum_climate_analysis()
```

## 🎯 Bài tập thực hành

### Bài tập 1: Protein Folding Simulation
```python
def quantum_protein_folding():
    """
    Mô phỏng protein folding sử dụng quantum algorithms
    """
    # TODO: Implement protein folding simulation
    pass
```

### Bài tập 2: Drug Discovery Pipeline
```python
def quantum_drug_discovery():
    """
    Pipeline khám phá thuốc sử dụng quantum simulation
    """
    # TODO: Implement drug discovery pipeline
    pass
```

### Bài tập 3: Quantum Machine Learning for Materials
```python
def quantum_ml_materials():
    """
    Machine learning lượng tử cho vật liệu
    """
    # TODO: Implement quantum ML for materials
    pass
```

## 📚 Tài liệu tham khảo

1. **Quantum Chemistry**: McArdle, S., Endo, S., Aspuru-Guzik, A. et al. (2020). Quantum computational chemistry.
2. **Material Science**: Ceperley, D.M. & Alder, B.J. (1980). Ground state of the electron gas by a stochastic method.
3. **Optimization**: Farhi, E., Goldstone, J. & Gutmann, S. (2014). A quantum approximate optimization algorithm.
4. **Financial Applications**: Orús, R., Mugel, S. & Lizaso, E. (2019). Quantum computing for finance: Overview and prospects.

## 🔮 Hướng dẫn tiếp theo

- **Day 34**: Quantum Computing on Real Hardware
- **Day 35**: Capstone Project và Portfolio Building

---

*"Quantum simulation will be one of the most important applications of quantum computers."* - Richard Feynman 