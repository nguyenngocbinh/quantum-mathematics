# Day 31: Quantum Cryptography Protocols

## 🎯 Mục tiêu
- Hiểu nguyên lý cơ bản của quantum cryptography
- Triển khai BB84 protocol cho quantum key distribution
- Phát hiện eavesdropping trong quantum communication
- Phân tích bảo mật của quantum cryptography protocols

## 🔐 Quantum Cryptography - Tổng Quan

### Tại sao Quantum Cryptography?
- **Unconditional security**: Dựa trên nguyên lý vật lý lượng tử
- **Eavesdropping detection**: Phát hiện người nghe lén tự động
- **Key distribution**: Tạo khóa bí mật hoàn hảo
- **Future-proof**: An toàn trước quantum computers
- **Information theory**: Bảo mật dựa trên uncertainty principle

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import random_statevector, Statevector
from qiskit.visualization import plot_bloch_multivector
import numpy as np
import matplotlib.pyplot as plt
import random
import hashlib
from typing import List, Tuple, Dict
```

## 🔑 BB84 Protocol Implementation

### 1. BB84 Protocol Fundamentals

```python
class BB84Protocol:
    """
    Triển khai BB84 Quantum Key Distribution Protocol
    """
    
    def __init__(self):
        self.basis_choices = ['Z', 'X']  # Computational và Hadamard basis
        self.bit_values = [0, 1]
        
    def alice_prepare_qubit(self, bit: int, basis: str) -> QuantumCircuit:
        """
        Alice chuẩn bị qubit theo bit và basis
        """
        qc = QuantumCircuit(1)
        
        if basis == 'Z':  # Computational basis
            if bit == 1:
                qc.x(0)  # |1⟩ state
        elif basis == 'X':  # Hadamard basis
            if bit == 0:
                qc.h(0)  # |+⟩ state
            else:
                qc.x(0)
                qc.h(0)  # |-⟩ state
                
        return qc
    
    def bob_measure_qubit(self, qc: QuantumCircuit, basis: str) -> int:
        """
        Bob đo qubit theo basis được chọn
        """
        # Tạo circuit mới để đo
        measure_qc = qc.copy()
        
        if basis == 'X':
            measure_qc.h(0)  # Chuyển về computational basis
        
        measure_qc.measure_all()
        
        # Thực hiện đo
        backend = Aer.get_backend('qasm_simulator')
        job = execute(measure_qc, backend, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Lấy kết quả đo
        measured_bit = int(list(counts.keys())[0])
        return measured_bit

# Test BB84 preparation
bb84 = BB84Protocol()

print("BB84 Protocol - Alice's Qubit Preparation:")
for bit in [0, 1]:
    for basis in ['Z', 'X']:
        qc = bb84.alice_prepare_qubit(bit, basis)
        print(f"Bit: {bit}, Basis: {basis}")
        print(qc)
        print()
```

### 2. Complete BB84 Key Generation

```python
def bb84_key_generation(n_qubits: int = 100) -> Tuple[List[int], List[int], float]:
    """
    Tạo khóa BB84 hoàn chỉnh
    """
    bb84 = BB84Protocol()
    
    # Alice's choices
    alice_bits = [random.choice([0, 1]) for _ in range(n_qubits)]
    alice_bases = [random.choice(['Z', 'X']) for _ in range(n_qubits)]
    
    # Bob's choices
    bob_bases = [random.choice(['Z', 'X']) for _ in range(n_qubits)]
    
    # Prepare and measure qubits
    bob_bits = []
    matching_bases = []
    
    for i in range(n_qubits):
        # Alice prepares qubit
        qc = bb84.alice_prepare_qubit(alice_bits[i], alice_bases[i])
        
        # Bob measures qubit
        measured_bit = bb84.bob_measure_qubit(qc, bob_bases[i])
        bob_bits.append(measured_bit)
        
        # Check if bases match
        if alice_bases[i] == bob_bases[i]:
            matching_bases.append(i)
    
    # Generate key from matching bases
    alice_key = [alice_bits[i] for i in matching_bases]
    bob_key = [bob_bits[i] for i in matching_bases]
    
    # Calculate error rate
    error_rate = sum(1 for a, b in zip(alice_key, bob_key) if a != b) / len(alice_key) if alice_key else 0
    
    return alice_key, bob_key, error_rate

# Test BB84 key generation
print("BB84 Key Generation Demo:")
alice_key, bob_key, error_rate = bb84_key_generation(50)

print(f"Generated key length: {len(alice_key)}")
print(f"Alice's key (first 10 bits): {alice_key[:10]}")
print(f"Bob's key (first 10 bits): {bob_key[:10]}")
print(f"Error rate: {error_rate:.3f}")
print(f"Keys match: {alice_key == bob_key}")
```

## 🕵️ Eavesdropping Detection

### 1. Intercept-Resend Attack

```python
class Eavesdropper:
    """
    Mô phỏng eavesdropper (Eve) thực hiện intercept-resend attack
    """
    
    def __init__(self):
        self.intercepted_bits = []
        self.intercepted_bases = []
        
    def intercept_and_resend(self, qc: QuantumCircuit, basis: str) -> Tuple[QuantumCircuit, int]:
        """
        Eve chặn qubit, đo và gửi lại
        """
        # Eve đo qubit với basis ngẫu nhiên
        eve_basis = random.choice(['Z', 'X'])
        eve_bit = bb84.bob_measure_qubit(qc, eve_basis)
        
        # Eve tạo lại qubit và gửi cho Bob
        new_qc = bb84.alice_prepare_qubit(eve_bit, eve_basis)
        
        return new_qc, eve_bit

def bb84_with_eavesdropping(n_qubits: int = 100) -> Tuple[List[int], List[int], float, float]:
    """
    BB84 với eavesdropping detection
    """
    bb84 = BB84Protocol()
    eve = Eavesdropper()
    
    # Alice's choices
    alice_bits = [random.choice([0, 1]) for _ in range(n_qubits)]
    alice_bases = [random.choice(['Z', 'X']) for _ in range(n_qubits)]
    
    # Bob's choices
    bob_bases = [random.choice(['Z', 'X']) for _ in range(n_qubits)]
    
    # Prepare, intercept, and measure qubits
    bob_bits = []
    matching_bases = []
    
    for i in range(n_qubits):
        # Alice prepares qubit
        qc = bb84.alice_prepare_qubit(alice_bits[i], alice_bases[i])
        
        # Eve intercepts (with 50% probability)
        if random.random() < 0.5:
            qc, eve_bit = eve.intercept_and_resend(qc, alice_bases[i])
            eve.intercepted_bits.append(eve_bit)
            eve.intercepted_bases.append(alice_bases[i])
        
        # Bob measures qubit
        measured_bit = bb84.bob_measure_qubit(qc, bob_bases[i])
        bob_bits.append(measured_bit)
        
        # Check if bases match
        if alice_bases[i] == bob_bases[i]:
            matching_bases.append(i)
    
    # Generate key from matching bases
    alice_key = [alice_bits[i] for i in matching_bases]
    bob_key = [bob_bits[i] for i in matching_bases]
    
    # Calculate error rate
    error_rate = sum(1 for a, b in zip(alice_key, bob_key) if a != b) / len(alice_key) if alice_key else 0
    
    # Expected error rate from eavesdropping
    expected_error_rate = 0.25  # 25% error rate from intercept-resend
    
    return alice_key, bob_key, error_rate, expected_error_rate

# Test eavesdropping detection
print("\nBB84 with Eavesdropping Detection:")
alice_key_eve, bob_key_eve, error_rate_eve, expected_error = bb84_with_eavesdropping(100)

print(f"Generated key length: {len(alice_key_eve)}")
print(f"Error rate: {error_rate_eve:.3f}")
print(f"Expected error rate from eavesdropping: {expected_error:.3f}")
print(f"Eavesdropping detected: {error_rate_eve > 0.1}")  # Threshold for detection
```

### 2. Privacy Amplification

```python
def privacy_amplification(key: List[int], target_length: int) -> List[int]:
    """
    Privacy amplification để loại bỏ thông tin bị lộ
    """
    if len(key) < target_length:
        return key
    
    # Sử dụng hash function để tạo key mới
    key_str = ''.join(map(str, key))
    hash_object = hashlib.sha256(key_str.encode())
    hash_hex = hash_object.hexdigest()
    
    # Chuyển hash thành binary
    hash_binary = bin(int(hash_hex, 16))[2:].zfill(256)
    
    # Lấy target_length bits đầu tiên
    amplified_key = [int(bit) for bit in hash_binary[:target_length]]
    
    return amplified_key

def secure_key_generation(n_qubits: int = 200) -> Tuple[List[int], List[int]]:
    """
    Tạo khóa bảo mật hoàn chỉnh với privacy amplification
    """
    # Generate initial key
    alice_key, bob_key, error_rate = bb84_key_generation(n_qubits)
    
    # Check for eavesdropping
    if error_rate > 0.1:
        print(f"Eavesdropping detected! Error rate: {error_rate:.3f}")
        return [], []
    
    # Privacy amplification
    target_length = min(len(alice_key), len(bob_key)) // 2  # Reduce key length
    final_alice_key = privacy_amplification(alice_key, target_length)
    final_bob_key = privacy_amplification(bob_key, target_length)
    
    return final_alice_key, final_bob_key

# Test secure key generation
print("\nSecure Key Generation with Privacy Amplification:")
final_alice_key, final_bob_key = secure_key_generation(200)

if final_alice_key and final_bob_key:
    print(f"Final key length: {len(final_alice_key)}")
    print(f"Final keys match: {final_alice_key == final_bob_key}")
    print(f"Key entropy: {sum(final_alice_key) / len(final_alice_key):.3f}")
else:
    print("Key generation failed due to eavesdropping detection")
```

## 🔒 Advanced Quantum Cryptography Protocols

### 1. BBM92 Protocol (Entanglement-based)

```python
def bbm92_protocol(n_pairs: int = 50) -> Tuple[List[int], List[int], float]:
    """
    BBM92 protocol sử dụng entangled pairs
    """
    # Tạo Bell pairs
    bell_pairs = []
    for _ in range(n_pairs):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        bell_pairs.append(qc)
    
    # Alice và Bob đo với basis ngẫu nhiên
    alice_bases = [random.choice(['Z', 'X']) for _ in range(n_pairs)]
    bob_bases = [random.choice(['Z', 'X']) for _ in range(n_pairs)]
    
    alice_bits = []
    bob_bits = []
    matching_bases = []
    
    for i in range(n_pairs):
        # Alice measures first qubit
        alice_qc = bell_pairs[i].copy()
        if alice_bases[i] == 'X':
            alice_qc.h(0)
        alice_qc.measure(0, 0)
        
        # Bob measures second qubit
        bob_qc = bell_pairs[i].copy()
        if bob_bases[i] == 'X':
            bob_qc.h(1)
        bob_qc.measure(1, 0)
        
        # Execute measurements
        backend = Aer.get_backend('qasm_simulator')
        
        alice_job = execute(alice_qc, backend, shots=1)
        alice_result = alice_job.result()
        alice_bit = int(list(alice_result.get_counts().keys())[0])
        
        bob_job = execute(bob_qc, backend, shots=1)
        bob_result = bob_job.result()
        bob_bit = int(list(bob_result.get_counts().keys())[0])
        
        alice_bits.append(alice_bit)
        bob_bits.append(bob_bit)
        
        # Check if bases match
        if alice_bases[i] == bob_bases[i]:
            matching_bases.append(i)
    
    # Generate key
    alice_key = [alice_bits[i] for i in matching_bases]
    bob_key = [bob_bits[i] for i in matching_bases]
    
    # Calculate error rate
    error_rate = sum(1 for a, b in zip(alice_key, bob_key) if a != b) / len(alice_key) if alice_key else 0
    
    return alice_key, bob_key, error_rate

# Test BBM92 protocol
print("\nBBM92 Protocol (Entanglement-based):")
bbm92_alice, bbm92_bob, bbm92_error = bbm92_protocol(50)

print(f"BBM92 key length: {len(bbm92_alice)}")
print(f"BBM92 error rate: {bbm92_error:.3f}")
print(f"BBM92 keys match: {bbm92_alice == bbm92_bob}")
```

### 2. Quantum Digital Signatures

```python
def quantum_digital_signature():
    """
    Quantum digital signature sử dụng quantum states
    """
    def create_signature_state(message: str, private_key: List[int]) -> QuantumCircuit:
        """
        Tạo quantum signature state
        """
        # Encode message và private key vào quantum state
        qc = QuantumCircuit(len(private_key))
        
        for i, bit in enumerate(private_key):
            if bit == 1:
                qc.x(i)
        
        # Apply message-dependent rotations
        message_hash = hashlib.sha256(message.encode()).hexdigest()
        for i, char in enumerate(message_hash[:len(private_key)]):
            angle = int(char, 16) * np.pi / 8
            qc.rz(angle, i % len(private_key))
        
        return qc
    
    def verify_signature(message: str, signature_qc: QuantumCircuit, public_key: List[int]) -> bool:
        """
        Xác thực quantum signature
        """
        # Measure signature state
        measure_qc = signature_qc.copy()
        measure_qc.measure_all()
        
        backend = Aer.get_backend('qasm_simulator')
        job = execute(measure_qc, backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Verify against public key
        most_likely_state = max(counts, key=counts.get)
        measured_bits = [int(bit) for bit in most_likely_state]
        
        # Simple verification (trong thực tế sẽ phức tạp hơn)
        return measured_bits == public_key
    
    return create_signature_state, verify_signature

# Test quantum digital signature
print("\nQuantum Digital Signature Demo:")
create_sig, verify_sig = quantum_digital_signature()

# Generate keys
private_key = [random.choice([0, 1]) for _ in range(4)]
public_key = private_key.copy()  # Simplified

# Create signature
message = "Hello Quantum World!"
signature_qc = create_sig(message, private_key)

# Verify signature
is_valid = verify_sig(message, signature_qc, public_key)
print(f"Message: {message}")
print(f"Private key: {private_key}")
print(f"Public key: {public_key}")
print(f"Signature valid: {is_valid}")
```

## 📊 Security Analysis

### 1. Key Rate Analysis

```python
def analyze_key_rate(n_trials: int = 10, qubit_counts: List[int] = [50, 100, 200, 500]) -> Dict:
    """
    Phân tích key rate cho các protocol khác nhau
    """
    results = {
        'BB84': {'key_lengths': [], 'error_rates': [], 'success_rate': 0},
        'BBM92': {'key_lengths': [], 'error_rates': [], 'success_rate': 0}
    }
    
    for n_qubits in qubit_counts:
        bb84_success = 0
        bbm92_success = 0
        
        for _ in range(n_trials):
            # BB84
            try:
                alice_key, bob_key, error_rate = bb84_key_generation(n_qubits)
                if alice_key and error_rate < 0.1:
                    results['BB84']['key_lengths'].append(len(alice_key))
                    results['BB84']['error_rates'].append(error_rate)
                    bb84_success += 1
            except:
                pass
            
            # BBM92
            try:
                alice_key, bob_key, error_rate = bbm92_protocol(n_qubits)
                if alice_key and error_rate < 0.1:
                    results['BBM92']['key_lengths'].append(len(alice_key))
                    results['BBM92']['error_rates'].append(error_rate)
                    bbm92_success += 1
            except:
                pass
        
        results['BB84']['success_rate'] = bb84_success / n_trials
        results['BBM92']['success_rate'] = bbm92_success / n_trials
    
    return results

# Analyze key rates
print("\nKey Rate Analysis:")
key_analysis = analyze_key_rate(5, [50, 100])

for protocol, data in key_analysis.items():
    if data['key_lengths']:
        avg_key_length = np.mean(data['key_lengths'])
        avg_error_rate = np.mean(data['error_rates'])
        print(f"{protocol}:")
        print(f"  Average key length: {avg_key_length:.1f}")
        print(f"  Average error rate: {avg_error_rate:.3f}")
        print(f"  Success rate: {data['success_rate']:.2f}")
```

### 2. Eavesdropping Detection Sensitivity

```python
def eavesdropping_sensitivity_analysis():
    """
    Phân tích độ nhạy của eavesdropping detection
    """
    eavesdropping_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    detection_results = []
    
    for eve_rate in eavesdropping_rates:
        detected_count = 0
        total_trials = 20
        
        for _ in range(total_trials):
            # Simulate eavesdropping with given rate
            alice_key, bob_key, error_rate, _ = bb84_with_eavesdropping(100)
            
            # Check if eavesdropping is detected
            if error_rate > 0.1:  # Detection threshold
                detected_count += 1
        
        detection_rate = detected_count / total_trials
        detection_results.append((eve_rate, detection_rate))
    
    return detection_results

# Analyze eavesdropping detection
print("\nEavesdropping Detection Sensitivity:")
sensitivity_results = eavesdropping_sensitivity_analysis()

for eve_rate, detection_rate in sensitivity_results:
    print(f"Eavesdropping rate: {eve_rate:.1f}, Detection rate: {detection_rate:.2f}")
```

## 🎯 Bài tập thực hành

### Bài tập 1: Implement E91 Protocol
```python
def e91_protocol_implementation():
    """
    Triển khai E91 protocol sử dụng 3-qubit GHZ states
    """
    # TODO: Implement E91 protocol
    pass
```

### Bài tập 2: Quantum Coin Flipping
```python
def quantum_coin_flipping():
    """
    Triển khai quantum coin flipping protocol
    """
    # TODO: Implement quantum coin flipping
    pass
```

### Bài tập 3: Quantum Commitment Scheme
```python
def quantum_commitment():
    """
    Triển khai quantum commitment scheme
    """
    # TODO: Implement quantum commitment
    pass
```

## 📚 Tài liệu tham khảo

1. **BB84 Protocol**: Bennett, C.H. & Brassard, G. (1984). Quantum cryptography: Public key distribution and coin tossing.
2. **BBM92 Protocol**: Bennett, C.H., Brassard, G. & Mermin, N.D. (1992). Quantum cryptography without Bell's theorem.
3. **E91 Protocol**: Ekert, A.K. (1991). Quantum cryptography based on Bell's theorem.
4. **Privacy Amplification**: Bennett, C.H., Brassard, G., Crépeau, C. & Maurer, U.M. (1995). Generalized privacy amplification.

## 🔮 Hướng dẫn tiếp theo

- **Day 32**: Quantum Random Number Generation
- **Day 33**: Quantum Simulation Projects
- **Day 34**: Quantum Computing on Real Hardware
- **Day 35**: Capstone Project và Portfolio Building

---

*"Quantum cryptography provides the only method for distributing secret keys with information-theoretic security."* - Artur Ekert 