

# 🌟 **Ngày 10: Toán tử lượng tử và Biến đổi Trạng thái (Operator & State Transformation)**

---

## **1. Toán tử lượng tử là gì?**

Trong toán học lượng tử:

✅ Toán tử (Operator) là phép biến đổi tác động lên trạng thái lượng tử
✅ Giống như phép nhân ma trận với vector trong đại số tuyến tính
✅ Giúp thay đổi trạng thái, tính toán, hoặc đo đạc hệ lượng tử

**Toán tử thường ký hiệu là chữ cái in hoa**, ví dụ:

| Toán tử           | Ý nghĩa                                      |     |
| ----------------- | -------------------------------------------- | --- |
| $X$               | Đảo bit (cổng NOT)                           |     |
| $H$               | Tạo siêu vị (Hadamard)                       |     |
| $Z$               | Đổi dấu trạng thái (                         | 1⟩) |
| Toán tử Hermitian | Dùng cho đo đạc, trị riêng là giá trị vật lý |     |

---

## **2. Toán tử Pauli-X (Đảo bit) cụ thể**

**Biểu diễn ma trận:**

$$
X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

**Tác động lên các trạng thái cơ bản:**

* Với trạng thái $|0⟩ = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$:

$$
X \, |0⟩ = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \times \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix} = |1⟩
$$

* Với trạng thái $|1⟩ = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$:

$$
X \, |1⟩ = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \times \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} = |0⟩
$$

**Kết luận:** Toán tử $X$ có tác dụng hoán đổi $|0⟩$ và $|1⟩$, tương tự cổng NOT trong máy tính cổ điển.

---

## **3. Biến đổi trạng thái siêu vị**

Giả sử hệ đang ở trạng thái siêu vị tổng quát:

$$
|\psi⟩ = a \, |0⟩ + b \, |1⟩
$$

Tác động toán tử $X$:

$$
X \, |\psi⟩ = a \, X|0⟩ + b \, X|1⟩ = a \, |1⟩ + b \, |0⟩ = b \, |0⟩ + a \, |1⟩
$$

**Ý nghĩa:** Hoán đổi hệ số của $|0⟩$ và $|1⟩$, trạng thái vẫn chuẩn hóa nếu ban đầu chuẩn hóa.

---

## **4. Bài tập mẫu cụ thể**

Cho trạng thái lượng tử:

$$
|\psi⟩ = \sqrt{0.7} \, |0⟩ + \sqrt{0.3} \, |1⟩
$$

**Bước 1: Kiểm tra chuẩn hóa**

Tính:

$$
|a|^2 + |b|^2 = (\sqrt{0.7})^2 + (\sqrt{0.3})^2 = 0.7 + 0.3 = 1
$$

✅ Chuẩn hóa hợp lệ.

---

**Bước 2: Tác động toán tử $X$**

Tính:

$$
X \, |\psi⟩ = \sqrt{0.7} \, X|0⟩ + \sqrt{0.3} \, X|1⟩ = \sqrt{0.7} \, |1⟩ + \sqrt{0.3} \, |0⟩
$$

Sắp xếp lại:

$$
|\psi'⟩ = \sqrt{0.3} \, |0⟩ + \sqrt{0.7} \, |1⟩
$$

---

**Bước 3: Kiểm tra chuẩn hóa sau biến đổi**

Tính:

$$
|a'|^2 + |b'|^2 = (\sqrt{0.3})^2 + (\sqrt{0.7})^2 = 0.3 + 0.7 = 1
$$

✅ Chuẩn hóa hợp lệ, trạng thái sau biến đổi là hợp pháp.

---

## **5. Tổng kết ý nghĩa**

* Toán tử lượng tử là công cụ biến đổi trạng thái
* Toán tử $X$ đảo bit lượng tử, cực kỳ cơ bản trong máy tính lượng tử
* Sau biến đổi, trạng thái vẫn giữ tổng xác suất bằng 1

---

# 🎯 **Bạn cần nhớ Ngày 10:**

✅ Trạng thái lượng tử mô tả bằng vector
✅ Toán tử lượng tử mô tả bằng ma trận, tác động lên vector trạng thái
✅ Các toán tử phổ biến như $X$, $H$, $Z$, toán tử đo đạc là nền tảng trong lượng tử học


---

# 🧩 **Bài tập Ngày 10 - Toán tử lượng tử và Biến đổi Trạng thái**

---

## **Đề bài:**

Cho trạng thái lượng tử:

$$
|\psi⟩ = \sqrt{0.7} \, |0⟩ + \sqrt{0.3} \, |1⟩
$$

Biết rằng toán tử Pauli-X được định nghĩa:

$$
X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

**Yêu cầu:**

1. Kiểm tra trạng thái $|\psi⟩$ có chuẩn hóa không
2. Tác động toán tử $X$ lên trạng thái $|\psi⟩$ và viết rõ trạng thái sau biến đổi
3. Kiểm tra trạng thái sau biến đổi có còn chuẩn hóa không
4. Giải thích kết quả

---

# ✅ **Giải bài tập từng bước**

---

## **Bước 1: Kiểm tra chuẩn hóa trạng thái ban đầu**

Trạng thái ban đầu:

$$
|\psi⟩ = \sqrt{0.7} \, |0⟩ + \sqrt{0.3} \, |1⟩
$$

Điều kiện chuẩn hóa:

$$
|a|^2 + |b|^2 = (\sqrt{0.7})^2 + (\sqrt{0.3})^2 = 0.7 + 0.3 = 1
$$

✅ Trạng thái đã được chuẩn hóa đúng.

---

## **Bước 2: Tác động toán tử $X$ lên $|\psi⟩$**

Nhớ rằng:

$$
X |0⟩ = |1⟩, \quad X |1⟩ = |0⟩
$$

Áp dụng:

$$
X \, |\psi⟩ = \sqrt{0.7} \, X|0⟩ + \sqrt{0.3} \, X|1⟩ = \sqrt{0.7} \, |1⟩ + \sqrt{0.3} \, |0⟩
$$

Sắp xếp lại cho đúng thứ tự:

$$
|\psi'⟩ = \sqrt{0.3} \, |0⟩ + \sqrt{0.7} \, |1⟩
$$

---

## **Bước 3: Kiểm tra chuẩn hóa trạng thái sau biến đổi**

Tính:

$$
|a'|^2 + |b'|^2 = (\sqrt{0.3})^2 + (\sqrt{0.7})^2 = 0.3 + 0.7 = 1
$$

✅ Trạng thái sau biến đổi vẫn chuẩn hóa đúng.

---

## **Bước 4: Giải thích kết quả**

* Toán tử $X$ có tác dụng đảo trạng thái cơ bản, giống như cổng NOT
* Trạng thái sau biến đổi vẫn là trạng thái hợp pháp (được chuẩn hóa)
* Tổng xác suất đo được các trạng thái vẫn bằng 1
* Đây là một ví dụ đơn giản về cách toán tử lượng tử thay đổi trạng thái mà không làm mất tính chuẩn hóa

---

# 🎯 **Kết luận Ngày 10:**

✅ Toán tử lượng tử có thể làm biến đổi trạng thái theo quy luật toán học rõ ràng
✅ Phép biến đổi không phá vỡ tính chuẩn hóa của hệ
✅ Đây là nền tảng để xây dựng các thuật toán và phép đo trong máy tính lượng tử


