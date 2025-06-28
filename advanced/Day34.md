# Day 34: Quantum Computing on Real Hardware

## 🎯 Mục tiêu
- Kết nối và sử dụng IBM Quantum Experience
- Hiểu về noise và decoherence trong quantum hardware
- Triển khai error mitigation techniques
- So sánh simulator vs real quantum hardware
- Tối ưu hóa circuits cho NISQ devices

## 🧠 Real Quantum Hardware - Tổng Quan

### Tại sao cần Real Hardware?
- **Noise characterization**: Hiểu thực tế về quantum noise
- **Decoherence effects**: Tác động của môi trường lên qubits
- **Error rates**: Tỷ lệ lỗi thực tế của các cổng lượng tử
- **NISQ limitations**: Giới hạn của Noisy Intermediate-Scale Quantum
- **Cloud access**: Truy cập quantum computers qua cloud

```python
from qiskit import QuantumCircuit, Aer, execute, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_error_map
from qiskit.ignis.mitigation import complete_meas_cal, CompleteMeasFitter
from qiskit.ignis.mitigation import measurement_calibration
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```

## 🔧 IBM Quantum Experience Setup

### 1. Kết nối IBM Quantum

```python
def setup_ibm_quantum():
    """
    Thiết lập kết nối với IBM Quantum Experience
    """
    try:
        # Load IBM Quantum account
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        
        # Get available backends
        backends = provider.backends()
        
        print("Available IBM Quantum Backends:")
        for backend in backends:
            if backend.configuration().simulator == False:
                print(f"- {backend.name()}: {backend.configuration().n_qubits} qubits")
        
        return provider, backends
    
    except Exception as e:
        print(f"Error connecting to IBM Quantum: {e}")
        print("Using local simulator instead")
        return None, None

def get_least_busy_backend(provider, min_qubits=2):
    """
    Lấy backend ít bận nhất với số qubit tối thiểu
    """
    if provider is None:
        return Aer.get_backend('qasm_simulator')
    
    backend = least_busy(provider.backends(
        filters=lambda x: x.configuration().n_qubits >= min_qubits 
        and not x.configuration().simulator
    ))
    
    print(f"Selected backend: {backend.name()}")
    return backend

# Setup IBM Quantum
provider, backends = setup_ibm_quantum()
```

### 2. Backend Information và Properties

```python
def analyze_backend_properties(backend):
    """
    Phân tích thuộc tính của quantum backend
    """
    print(f"\n=== Backend Analysis: {backend.name()} ===")
    
    # Configuration
    config = backend.configuration()
    print(f"Number of qubits: {config.n_qubits}")
    print(f"Max experiments: {config.max_experiments}")
    print(f"Max shots: {config.max_shots}")
    print(f"Simulator: {config.simulator}")
    
    # Properties (if available)
    try:
        properties = backend.properties()
        
        # Gate errors
        print("\nGate Errors:")
        for gate in properties.gates:
            if gate.gate == 'cx':
                for qubit in gate.qubits:
                    error = gate.parameters[0].value
                    print(f"  CX({qubit[0]}, {qubit[1]}): {error:.4f}")
        
        # Qubit properties
        print("\nQubit Properties:")
        for i in range(min(5, config.n_qubits)):  # Show first 5 qubits
            t1 = properties.qubits[i][0].value
            t2 = properties.qubits[i][1].value
            readout_error = properties.qubits[i][2].value
            print(f"  Qubit {i}: T1={t1:.2f}μs, T2={t2:.2f}μs, Readout Error={readout_error:.4f}")
    
    except Exception as e:
        print(f"Could not retrieve properties: {e}")
    
    return config

# Analyze backend
if provider:
    backend = get_least_busy_backend(provider)
    config = analyze_backend_properties(backend)
else:
    backend = Aer.get_backend('qasm_simulator')
    print("Using local simulator")
```

## 🧪 Noise và Error Characterization

### 1. Noise Model Simulation

```python
def create_noise_model():
    """
    Tạo noise model để mô phỏng real hardware
    """
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer.noise.errors import depolarizing_error, thermal_relaxation_error
    
    # Create noise model
    noise_model = NoiseModel()
    
    # Add depolarizing error to single-qubit gates
    error_1q = depolarizing_error(0.001, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3'])
    
    # Add depolarizing error to two-qubit gates
    error_2q = depolarizing_error(0.01, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    
    # Add thermal relaxation error
    t1 = 50.0  # microseconds
    t2 = 70.0  # microseconds
    error_relax = thermal_relaxation_error(t1, t2, 0.1)  # 0.1μs gate time
    noise_model.add_all_qubit_quantum_error(error_relax, ['u1', 'u2', 'u3'])
    
    return noise_model

def compare_simulator_vs_noise():
    """
    So sánh kết quả giữa ideal simulator và noisy simulator
    """
    # Create Bell state circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    # Ideal simulation
    backend_ideal = Aer.get_backend('qasm_simulator')
    job_ideal = execute(qc, backend_ideal, shots=1000)
    result_ideal = job_ideal.result()
    counts_ideal = result_ideal.get_counts()
    
    # Noisy simulation
    noise_model = create_noise_model()
    backend_noisy = Aer.get_backend('qasm_simulator')
    job_noisy = execute(qc, backend_noisy, shots=1000, noise_model=noise_model)
    result_noisy = job_noisy.result()
    counts_noisy = result_noisy.get_counts()
    
    print("Bell State Results:")
    print(f"Ideal simulator: {counts_ideal}")
    print(f"Noisy simulator: {counts_noisy}")
    
    return counts_ideal, counts_noisy

# Compare simulations
ideal_counts, noisy_counts = compare_simulator_vs_noise()
```

### 2. Error Mitigation Techniques

```python
def measurement_error_mitigation():
    """
    Triển khai measurement error mitigation
    """
    # Create calibration circuits
    meas_calibs, state_labels = complete_meas_cal(
        qubit_list=[0, 1], 
        circlabel='mcal'
    )
    
    # Create test circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    # Execute calibration and test circuits
    backend = Aer.get_backend('qasm_simulator')
    
    # Add noise to calibration
    noise_model = create_noise_model()
    
    # Execute calibration
    cal_job = execute(meas_calibs, backend, shots=1000, noise_model=noise_model)
    cal_results = cal_job.result()
    
    # Execute test circuit
    test_job = execute(qc, backend, shots=1000, noise_model=noise_model)
    test_results = test_job.result()
    
    # Create measurement filter
    meas_fitter = CompleteMeasFitter(cal_results, state_labels)
    
    # Apply correction
    mitigated_results = meas_fitter.filter.apply(test_results)
    
    print("Measurement Error Mitigation Results:")
    print(f"Original counts: {test_results.get_counts()}")
    print(f"Mitigated counts: {mitigated_results.get_counts()}")
    
    return meas_fitter, test_results, mitigated_results

# Error mitigation demo
meas_fitter, original_results, mitigated_results = measurement_error_mitigation()
```

## 🚀 Real Hardware Execution

### 1. Circuit Optimization cho NISQ

```python
def optimize_circuit_for_nisq(qc):
    """
    Tối ưu hóa circuit cho NISQ devices
    """
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Optimize1qGates, CXCancellation
    
    # Create pass manager for optimization
    pm = PassManager()
    pm.append(Optimize1qGates())
    pm.append(CXCancellation())
    
    # Optimize circuit
    optimized_qc = pm.run(qc)
    
    print(f"Original circuit depth: {qc.depth()}")
    print(f"Optimized circuit depth: {optimized_qc.depth()}")
    print(f"Original circuit gates: {qc.count_ops()}")
    print(f"Optimized circuit gates: {optimized_qc.count_ops()}")
    
    return optimized_qc

def create_nisq_friendly_circuit():
    """
    Tạo circuit thân thiện với NISQ devices
    """
    qc = QuantumCircuit(3, 3)
    
    # Use simple gates that are well-calibrated
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    
    return qc

# Create and optimize circuit
nisq_circuit = create_nisq_friendly_circuit()
optimized_circuit = optimize_circuit_for_nisq(nisq_circuit)
```

### 2. Real Hardware Execution

```python
def execute_on_real_hardware(qc, backend, shots=1000):
    """
    Thực thi circuit trên real quantum hardware
    """
    try:
        print(f"Submitting job to {backend.name()}...")
        
        # Submit job
        job = execute(qc, backend, shots=shots)
        
        # Monitor job
        job_monitor(job)
        
        # Get results
        result = job.result()
        counts = result.get_counts()
        
        print(f"Job completed! Results: {counts}")
        
        return result, counts
    
    except Exception as e:
        print(f"Error executing on real hardware: {e}")
        return None, None

def compare_all_backends():
    """
    So sánh kết quả trên các backend khác nhau
    """
    # Create test circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    results = {}
    
    # Ideal simulator
    backend_ideal = Aer.get_backend('qasm_simulator')
    job_ideal = execute(qc, backend_ideal, shots=1000)
    result_ideal = job_ideal.result()
    results['ideal'] = result_ideal.get_counts()
    
    # Noisy simulator
    noise_model = create_noise_model()
    backend_noisy = Aer.get_backend('qasm_simulator')
    job_noisy = execute(qc, backend_noisy, shots=1000, noise_model=noise_model)
    result_noisy = job_noisy.result()
    results['noisy'] = result_noisy.get_counts()
    
    # Real hardware (if available)
    if provider:
        try:
            real_result, real_counts = execute_on_real_hardware(qc, backend)
            if real_counts:
                results['real_hardware'] = real_counts
        except:
            print("Could not execute on real hardware")
    
    print("\n=== Comparison Results ===")
    for backend_name, counts in results.items():
        print(f"{backend_name}: {counts}")
    
    return results

# Compare all backends
comparison_results = compare_all_backends()
```

## 📊 Error Analysis và Visualization

### 1. Error Map Visualization

```python
def visualize_error_map(backend):
    """
    Trực quan hóa error map của backend
    """
    try:
        properties = backend.properties()
        
        # Create error map
        error_map = plot_error_map(properties)
        plt.show()
        
        return error_map
    
    except Exception as e:
        print(f"Could not create error map: {e}")
        return None

def analyze_error_statistics(results_dict):
    """
    Phân tích thống kê lỗi
    """
    print("\n=== Error Analysis ===")
    
    if 'ideal' in results_dict and 'noisy' in results_dict:
        ideal = results_dict['ideal']
        noisy = results_dict['noisy']
        
        # Calculate fidelity
        total_ideal = sum(ideal.values())
        total_noisy = sum(noisy.values())
        
        fidelity = 0
        for state in ideal:
            if state in noisy:
                fidelity += min(ideal[state], noisy[state])
        
        fidelity /= total_ideal
        print(f"Fidelity (ideal vs noisy): {fidelity:.4f}")
    
    return fidelity

# Visualize and analyze
if provider:
    error_map = visualize_error_map(backend)
fidelity = analyze_error_statistics(comparison_results)
```

## 🎯 Bài tập thực hành

### Bài tập 1: Noise Characterization
```python
def noise_characterization_exercise():
    """
    Bài tập: Phân tích noise trên real hardware
    """
    # 1. Tạo circuit đơn giản (Bell state)
    # 2. Chạy trên simulator và real hardware
    # 3. So sánh kết quả và tính fidelity
    # 4. Phân tích nguyên nhân lỗi
    
    pass

### Bài tập 2: Error Mitigation
```python
def error_mitigation_exercise():
    """
    Bài tập: Triển khai error mitigation
    """
    # 1. Implement measurement error mitigation
    # 2. So sánh kết quả trước và sau mitigation
    # 3. Đánh giá hiệu quả của mitigation
    
    pass

### Bài tập 3: Circuit Optimization
```python
def circuit_optimization_exercise():
    """
    Bài tập: Tối ưu hóa circuit cho NISQ
    """
    # 1. Tạo circuit phức tạp
    # 2. Áp dụng các kỹ thuật optimization
    # 3. Đo lường cải thiện performance
    
    pass
```

## 📚 Tài nguyên bổ sung

### IBM Quantum Resources:
- [IBM Quantum Experience](https://quantum-computing.ibm.com/)
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [IBM Quantum Learning](https://learning.quantum-computing.ibm.com/)

### Error Mitigation Papers:
- "Error Mitigation for Short-Depth Quantum Circuits" - Temme et al.
- "Scalable Error Mitigation for Noisy Quantum Circuits" - Kandala et al.

### NISQ Computing:
- "Quantum Computing in the NISQ era and beyond" - Preskill
- "Noisy intermediate-scale quantum algorithms" - Bharti et al.

---

## 🎯 Tổng kết ngày 34

### Kỹ năng đạt được:
- ✅ Kết nối và sử dụng IBM Quantum Experience
- ✅ Hiểu về noise và error trong quantum hardware
- ✅ Triển khai error mitigation techniques
- ✅ Tối ưu hóa circuits cho NISQ devices
- ✅ So sánh simulator vs real hardware

### Chuẩn bị cho ngày tiếp theo:
- Portfolio building và documentation
- Capstone project planning
- Career preparation trong quantum computing 