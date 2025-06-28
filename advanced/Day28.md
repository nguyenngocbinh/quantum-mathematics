# Day 28: Quantum Approximate Optimization Algorithm (QAOA)

## 🎯 Mục tiêu
- Hiểu nguyên lý hoạt động của QAOA
- Triển khai QAOA cho bài toán MaxCut
- Thiết kế cost function và parameter optimization
- Áp dụng hybrid quantum-classical optimization

## 🧠 QAOA - Tổng Quan

### Tại sao QAOA?
- **Combinatorial optimization**: Giải quyết các bài toán tối ưu hóa tổ hợp
- **Quantum advantage**: Tận dụng quantum superposition và entanglement
- **Hybrid approach**: Kết hợp quantum và classical optimization
- **Near-term quantum**: Phù hợp với NISQ (Noisy Intermediate-Scale Quantum) devices

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliSumOp, I, Z
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import minimize
```

## 🔧 QAOA Fundamentals

### 1. QAOA Circuit Structure

```python
def qaoa_circuit_demo():
    """
    Demo cấu trúc QAOA circuit cơ bản
    """
    # Parameters
    n_qubits = 4
    p = 2  # Number of layers
    
    # Create parameters
    gamma = Parameter('γ')
    beta = Parameter('β')
    
    # Create QAOA circuit
    qc = QuantumCircuit(n_qubits)
    
    # Initial state: |+⟩^⊗n
    for i in range(n_qubits):
        qc.h(i)
    
    # QAOA layers
    for layer in range(p):
        # Cost Hamiltonian layer (U_C)
        for i in range(n_qubits):
            qc.rz(gamma, i)
        
        # Mixing Hamiltonian layer (U_M)
        for i in range(n_qubits):
            qc.rx(beta, i)
    
    # Measure
    qc.measure_all()
    
    print("QAOA Circuit Structure:")
    print(qc)
    
    return qc

# Tạo QAOA circuit
qaoa_qc = qaoa_circuit_demo()
```

### 2. Cost Hamiltonian Construction

```python
def create_cost_hamiltonian(graph):
    """
    Tạo cost Hamiltonian cho MaxCut problem
    """
    # Cost function: maximize sum of edge weights across cut
    # H_C = -∑(i,j)∈E w_ij * Z_i * Z_j
    
    cost_operators = []
    
    for edge in graph.edges():
        i, j = edge
        weight = graph[i][j].get('weight', 1.0)
        
        # Create Z_i * Z_j operator
        pauli_string = ['I'] * max(i, j) + 1
        pauli_string[i] = 'Z'
        pauli_string[j] = 'Z'
        
        # Convert to PauliSumOp
        pauli_op = PauliSumOp.from_list([(''.join(pauli_string), -weight)])
        cost_operators.append(pauli_op)
    
    # Sum all operators
    cost_hamiltonian = sum(cost_operators)
    
    return cost_hamiltonian

def create_mixing_hamiltonian(n_qubits):
    """
    Tạo mixing Hamiltonian: H_M = -∑_i X_i
    """
    mixing_operators = []
    
    for i in range(n_qubits):
        pauli_string = ['I'] * n_qubits
        pauli_string[i] = 'X'
        
        pauli_op = PauliSumOp.from_list([(''.join(pauli_string), -1.0)])
        mixing_operators.append(pauli_op)
    
    mixing_hamiltonian = sum(mixing_operators)
    
    return mixing_hamiltonian

# Test với graph đơn giản
test_graph = nx.Graph()
test_graph.add_edge(0, 1, weight=1.0)
test_graph.add_edge(1, 2, weight=1.0)
test_graph.add_edge(2, 3, weight=1.0)
test_graph.add_edge(3, 0, weight=1.0)

cost_ham = create_cost_hamiltonian(test_graph)
mixing_ham = create_mixing_hamiltonian(4)

print("Cost Hamiltonian:")
print(cost_ham)
print("\nMixing Hamiltonian:")
print(mixing_ham)
```

## 🎯 MaxCut Problem với QAOA

### 1. MaxCut Problem Setup

```python
def maxcut_problem_demo():
    """
    Demo QAOA cho MaxCut problem
    """
    # Tạo graph
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(3, 0, weight=1.0)
    G.add_edge(0, 2, weight=0.5)
    
    # Visualize graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, font_weight='bold')
    
    # Draw edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title('MaxCut Problem Graph')
    plt.show()
    
    return G

# Tạo MaxCut problem
maxcut_graph = maxcut_problem_demo()
```

### 2. QAOA Implementation cho MaxCut

```python
def qaoa_maxcut_implementation(graph, p=2):
    """
    Triển khai QAOA cho MaxCut
    """
    n_qubits = len(graph.nodes())
    
    # Create cost and mixing Hamiltonians
    cost_hamiltonian = create_cost_hamiltonian(graph)
    mixing_hamiltonian = create_mixing_hamiltonian(n_qubits)
    
    # Create QAOA
    qaoa = QAOA(
        optimizer=COBYLA(maxiter=100),
        quantum_instance=Aer.get_backend('qasm_simulator'),
        reps=p
    )
    
    # Solve
    result = qaoa.solve(cost_hamiltonian)
    
    print(f"Optimal parameters: {result.optimal_parameters}")
    print(f"Optimal value: {result.optimal_value}")
    print(f"Optimal point: {result.optimal_point}")
    
    return result, qaoa

# Chạy QAOA cho MaxCut
qaoa_result, qaoa_algorithm = qaoa_maxcut_implementation(maxcut_graph)
```

### 3. Classical Solution Comparison

```python
def classical_maxcut_solution(graph):
    """
    Giải MaxCut bằng classical algorithm để so sánh
    """
    # Simple greedy algorithm
    nodes = list(graph.nodes())
    n = len(nodes)
    
    best_cut = 0
    best_partition = None
    
    # Try all possible partitions
    for i in range(2**(n-1)):  # Only need to check half due to symmetry
        partition = []
        for j in range(n):
            if (i >> j) & 1:
                partition.append(nodes[j])
        
        # Calculate cut value
        cut_value = 0
        for edge in graph.edges():
            u, v = edge
            if (u in partition) != (v in partition):  # Different sides
                cut_value += graph[u][v]['weight']
        
        if cut_value > best_cut:
            best_cut = cut_value
            best_partition = partition
    
    return best_cut, best_partition

# So sánh classical vs quantum
classical_cut, classical_partition = classical_maxcut_solution(maxcut_graph)
print(f"Classical MaxCut value: {classical_cut}")
print(f"Classical partition: {classical_partition}")
print(f"QAOA MaxCut value: {-qaoa_result.optimal_value}")  # Note: QAOA minimizes, so we negate
```

## 🔄 Parameter Optimization

### 1. Manual Parameter Optimization

```python
def manual_parameter_optimization(graph, p=2):
    """
    Tối ưu hóa parameters thủ công
    """
    n_qubits = len(graph.nodes())
    cost_hamiltonian = create_cost_hamiltonian(graph)
    
    def objective_function(params):
        # Split parameters into gamma and beta
        gamma = params[:p]
        beta = params[p:]
        
        # Create QAOA circuit
        qc = QuantumCircuit(n_qubits)
        
        # Initial state
        for i in range(n_qubits):
            qc.h(i)
        
        # QAOA layers
        for layer in range(p):
            # Cost layer
            for i in range(n_qubits):
                qc.rz(gamma[layer], i)
            
            # Mixing layer
            for i in range(n_qubits):
                qc.rx(beta[layer], i)
        
        qc.measure_all()
        
        # Execute
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate expectation value
        expectation = 0
        for bitstring, count in counts.items():
            # Convert bitstring to measurement result
            measurement = [int(bit) for bit in bitstring]
            
            # Calculate cost for this measurement
            cost = 0
            for edge in graph.edges():
                i, j = edge
                if measurement[i] != measurement[j]:  # Different sides
                    cost += graph[i][j]['weight']
            
            expectation += (count / 1000) * cost
        
        return -expectation  # Minimize negative of expectation
    
    # Initial parameters
    initial_params = np.random.rand(2*p) * 2 * np.pi
    
    # Optimize
    result = minimize(objective_function, initial_params, method='COBYLA')
    
    print(f"Optimal parameters: {result.x}")
    print(f"Optimal value: {-result.fun}")
    
    return result

# Chạy manual optimization
manual_result = manual_parameter_optimization(maxcut_graph)
```

### 2. Parameter Landscape Visualization

```python
def visualize_parameter_landscape(graph, p=1):
    """
    Trực quan hóa parameter landscape
    """
    n_qubits = len(graph.nodes())
    cost_hamiltonian = create_cost_hamiltonian(graph)
    
    def objective_2d(params):
        # For p=1, we have 2 parameters: gamma and beta
        gamma, beta = params
        
        # Create circuit
        qc = QuantumCircuit(n_qubits)
        
        # Initial state
        for i in range(n_qubits):
            qc.h(i)
        
        # Single QAOA layer
        for i in range(n_qubits):
            qc.rz(gamma, i)
        
        for i in range(n_qubits):
            qc.rx(beta, i)
        
        qc.measure_all()
        
        # Execute
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate expectation
        expectation = 0
        for bitstring, count in counts.items():
            measurement = [int(bit) for bit in bitstring]
            cost = 0
            for edge in graph.edges():
                i, j = edge
                if measurement[i] != measurement[j]:
                    cost += graph[i][j]['weight']
            expectation += (count / 1000) * cost
        
        return -expectation
    
    # Create parameter grid
    gamma_range = np.linspace(0, 2*np.pi, 20)
    beta_range = np.linspace(0, 2*np.pi, 20)
    gamma_grid, beta_grid = np.meshgrid(gamma_range, beta_range)
    
    # Calculate objective values
    Z = np.zeros_like(gamma_grid)
    for i in range(len(gamma_range)):
        for j in range(len(beta_range)):
            Z[i, j] = objective_2d([gamma_grid[i, j], beta_grid[i, j]])
    
    # Plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(gamma_grid, beta_grid, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Objective Value')
    plt.xlabel('γ (gamma)')
    plt.ylabel('β (beta)')
    plt.title('QAOA Parameter Landscape (p=1)')
    plt.show()
    
    return gamma_grid, beta_grid, Z

# Visualize parameter landscape
gamma_g, beta_g, obj_values = visualize_parameter_landscape(maxcut_graph)
```

## 📊 Performance Analysis

### 1. QAOA vs Classical Comparison

```python
def qaoa_performance_analysis():
    """
    Phân tích hiệu suất QAOA vs classical algorithms
    """
    # Test trên nhiều graph sizes
    graph_sizes = [4, 6, 8, 10]
    p_values = [1, 2, 3]
    
    results = {}
    
    for size in graph_sizes:
        # Create random graph
        G = nx.random_regular_graph(3, size)  # 3-regular graph
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = np.random.rand()
        
        # Classical solution
        classical_cut, _ = classical_maxcut_solution(G)
        
        results[size] = {'classical': classical_cut, 'qaoa': {}}
        
        for p in p_values:
            try:
                result, _ = qaoa_maxcut_implementation(G, p)
                qaoa_cut = -result.optimal_value
                results[size]['qaoa'][p] = qaoa_cut
            except:
                results[size]['qaoa'][p] = None
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    for size in graph_sizes:
        classical = results[size]['classical']
        qaoa_values = [results[size]['qaoa'].get(p, 0) for p in p_values]
        
        plt.subplot(2, 2, graph_sizes.index(size) + 1)
        plt.bar(['Classical'] + [f'QAOA p={p}' for p in p_values], 
                [classical] + qaoa_values)
        plt.title(f'Graph Size {size}')
        plt.ylabel('MaxCut Value')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Chạy performance analysis
performance_results = qaoa_performance_analysis()
```

### 2. Approximation Ratio Analysis

```python
def approximation_ratio_analysis():
    """
    Phân tích approximation ratio của QAOA
    """
    # Test trên nhiều instances
    n_instances = 20
    graph_size = 6
    
    approximation_ratios = []
    
    for i in range(n_instances):
        # Create random graph
        G = nx.random_regular_graph(3, graph_size)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = np.random.rand()
        
        # Classical optimal
        classical_cut, _ = classical_maxcut_solution(G)
        
        # QAOA solution
        try:
            result, _ = qaoa_maxcut_implementation(G, p=2)
            qaoa_cut = -result.optimal_value
            
            # Approximation ratio
            ratio = qaoa_cut / classical_cut
            approximation_ratios.append(ratio)
        except:
            continue
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(approximation_ratios, bins=10, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(approximation_ratios), color='red', linestyle='--', 
                label=f'Mean: {np.mean(approximation_ratios):.3f}')
    plt.xlabel('Approximation Ratio')
    plt.ylabel('Frequency')
    plt.title('QAOA Approximation Ratio Distribution')
    plt.legend()
    plt.show()
    
    print(f"Average approximation ratio: {np.mean(approximation_ratios):.3f}")
    print(f"Standard deviation: {np.std(approximation_ratios):.3f}")
    
    return approximation_ratios

# Chạy approximation ratio analysis
approx_ratios = approximation_ratio_analysis()
```

## 📚 Bài Tập Thực Hành

### Bài tập 1: QAOA cho Graph Coloring
```python
def qaoa_graph_coloring():
    """
    Áp dụng QAOA cho Graph Coloring problem
    """
    # Graph coloring: minimize number of colors needed
    # Cost function: penalize adjacent vertices with same color
    
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 0)
    G.add_edge(0, 2)
    
    # Create cost Hamiltonian for graph coloring
    # H_C = ∑(i,j)∈E δ(c_i, c_j) where δ is Kronecker delta
    
    n_qubits = len(G.nodes()) * 2  # 2 qubits per node for 4 colors
    
    def create_coloring_cost_hamiltonian(graph, n_colors=4):
        cost_operators = []
        
        for edge in graph.edges():
            i, j = edge
            
            # For each color, penalize if both vertices have same color
            for color in range(n_colors):
                # Create operator that checks if both vertices have same color
                pauli_string = ['I'] * (len(graph.nodes()) * n_colors)
                
                # Set qubits for color 'color' for both vertices
                pauli_string[i * n_colors + color] = 'Z'
                pauli_string[j * n_colors + color] = 'Z'
                
                # This operator gives +1 if both vertices have same color
                pauli_op = PauliSumOp.from_list([(''.join(pauli_string), 1.0)])
                cost_operators.append(pauli_op)
        
        return sum(cost_operators)
    
    cost_ham = create_coloring_cost_hamiltonian(G)
    
    # Implement QAOA for graph coloring
    # (This is a simplified version - full implementation would be more complex)
    
    print("Graph Coloring Cost Hamiltonian created")
    return G, cost_ham

# Chạy graph coloring QAOA
coloring_graph, coloring_cost = qaoa_graph_coloring()
```

### Bài tập 2: QAOA với Custom Mixing Hamiltonian
```python
def custom_mixing_hamiltonian_qaoa():
    """
    QAOA với custom mixing Hamiltonian
    """
    # Create custom mixing Hamiltonian: H_M = -∑_i (X_i + Y_i)
    
    def create_custom_mixing_hamiltonian(n_qubits):
        mixing_operators = []
        
        for i in range(n_qubits):
            # X operator
            pauli_string_x = ['I'] * n_qubits
            pauli_string_x[i] = 'X'
            pauli_op_x = PauliSumOp.from_list([(''.join(pauli_string_x), -1.0)])
            
            # Y operator
            pauli_string_y = ['I'] * n_qubits
            pauli_string_y[i] = 'Y'
            pauli_op_y = PauliSumOp.from_list([(''.join(pauli_string_y), -1.0)])
            
            mixing_operators.extend([pauli_op_x, pauli_op_y])
        
        return sum(mixing_operators)
    
    # Test với simple graph
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 0, weight=1.0)
    
    cost_ham = create_cost_hamiltonian(G)
    custom_mixing_ham = create_custom_mixing_hamiltonian(3)
    
    print("Custom mixing Hamiltonian created")
    print(custom_mixing_ham)
    
    return G, cost_ham, custom_mixing_ham

# Chạy custom mixing QAOA
custom_graph, custom_cost, custom_mixing = custom_mixing_hamiltonian_qaoa()
```

### Bài tập 3: QAOA Parameter Scheduling
```python
def qaoa_parameter_scheduling():
    """
    Tối ưu hóa parameter scheduling cho QAOA
    """
    # Implement different parameter initialization strategies
    
    def linear_schedule(p):
        """Linear parameter schedule"""
        gamma = np.linspace(0, 2*np.pi, p)
        beta = np.linspace(0, np.pi, p)
        return np.concatenate([gamma, beta])
    
    def random_schedule(p):
        """Random parameter initialization"""
        return np.random.rand(2*p) * 2 * np.pi
    
    def optimal_schedule(p):
        """Optimal parameter schedule based on theory"""
        # For MaxCut, optimal parameters follow specific patterns
        gamma = np.ones(p) * np.pi / 4
        beta = np.ones(p) * np.pi / 4
        return np.concatenate([gamma, beta])
    
    # Test different schedules
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(3, 0, weight=1.0)
    
    p = 3
    schedules = {
        'Linear': linear_schedule(p),
        'Random': random_schedule(p),
        'Optimal': optimal_schedule(p)
    }
    
    results = {}
    for name, params in schedules.items():
        print(f"Testing {name} schedule: {params}")
        # Here you would run QAOA with these parameters
        results[name] = params
    
    return results

# Chạy parameter scheduling
schedule_results = qaoa_parameter_scheduling()
```

## 🎯 Kết Quả Mong Đợi
- Hiểu rõ nguyên lý QAOA và ứng dụng cho optimization problems
- Có thể triển khai QAOA cho MaxCut và các bài toán khác
- Tối ưu hóa parameters hiệu quả
- So sánh hiệu suất với classical algorithms

## 📖 Tài Liệu Tham Khảo
- [QAOA Paper](https://arxiv.org/abs/1411.4028)
- [Qiskit QAOA](https://qiskit.org/documentation/stubs/qiskit.algorithms.QAOA.html)
- [MaxCut Problem](https://en.wikipedia.org/wiki/Maximum_cut)
- [Quantum Optimization](https://qiskit.org/textbook/ch-applications/qaoa.html) 