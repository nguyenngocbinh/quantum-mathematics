

# 🧩 **Ngày 4: Trị riêng (Eigenvalues) và Vector riêng (Eigenvectors)**

---

## **1. Khái niệm đơn giản nhất**

**Vector riêng** là một vector khi bị tác động bởi một ma trận sẽ **không đổi hướng**, chỉ thay đổi độ lớn.

**Trị riêng** là con số cho biết độ lớn bị thay đổi bao nhiêu.

---

### **Công thức cơ bản:**

Cho ma trận **A** và vector **v**, nếu:

$$
A \times \mathbf{v} = \lambda \times \mathbf{v}
$$

* **v** là **vector riêng**
* **λ** (lambda) là **trị riêng**
* Nghĩa là: Sau khi nhân ma trận A vào v, vector v vẫn cùng hướng, chỉ bị kéo dài hoặc rút ngắn bởi λ.

---

## **2. Ví dụ trực quan:**

Giả sử:

$$
A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}
$$

Vector:

$$
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

Tính:

$$
A \times \mathbf{v}_1 = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \times \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 0 \end{bmatrix} = 2 \times \mathbf{v}_1
$$

Vậy:

* **v₁** là vector riêng
* **λ = 2** là trị riêng tương ứng

---

## **3. Vì sao quan trọng trong lượng tử?**

Trong lượng tử:

* **Vector riêng** = Trạng thái ổn định khi đo đạc
* **Trị riêng** = Giá trị đo được (kết quả vật lý, ví dụ năng lượng, vị trí...)

Bạn chỉ đo được giá trị **trị riêng**, không thể biết toàn bộ trạng thái nếu chưa đo.

---

## **4. Bài tập tự làm cho bạn:**

Cho ma trận:

$$
B = \begin{bmatrix} 4 & 0 \\ 0 & 5 \end{bmatrix}
$$

1. Tìm vector riêng và trị riêng của ma trận B
2. Kiểm tra lại phép tính như ví dụ trên

**Gợi ý:** Thử với:

$$
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

---

