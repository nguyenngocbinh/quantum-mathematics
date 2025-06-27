
# 🌟 **Ngày 11: Toán tử Hadamard và Siêu vị mạnh**

---

## **1. Toán tử Hadamard là gì?**

Toán tử Hadamard, ký hiệu là:

$$
H = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
$$

Đây là một trong những toán tử quan trọng nhất trong lượng tử học, có khả năng:

✅ Biến trạng thái cơ bản thành trạng thái siêu vị (superposition)
✅ Là nền tảng cho tính toán song song của máy tính lượng tử

---

## **2. Tác động của toán tử Hadamard**

**Tác động lên $|0⟩$:**

$$
H \, |0⟩ = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \times \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

Vậy:

$$
H \, |0⟩ = \frac{1}{\sqrt{2}} \, |0⟩ + \frac{1}{\sqrt{2}} \, |1⟩
$$

✅ Đây là trạng thái siêu vị cân bằng, 50% $|0⟩$, 50% $|1⟩$.

---

**Tác động lên $|1⟩$:**

$$
H \, |1⟩ = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \times \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

Viết lại:

$$
H \, |1⟩ = \frac{1}{\sqrt{2}} \, |0⟩ - \frac{1}{\sqrt{2}} \, |1⟩
$$

Cũng là trạng thái siêu vị nhưng có sự khác biệt về dấu.

---

## **3. Ý nghĩa vật lý của toán tử Hadamard**

* Khi áp dụng lên $|0⟩$, tạo ra trạng thái siêu vị hoàn hảo — hạt vừa ở $|0⟩$ vừa ở $|1⟩$ với xác suất 50-50
* Là cơ sở để tạo ra sự "song song tính toán" trong thuật toán lượng tử
* Nếu áp dụng lần nữa, trạng thái sẽ trở về ban đầu (Hadamard là toán tử tự nghịch đảo)

---

## **4. Ví dụ thực tế**

Cho hệ đang ở $|0⟩$:

Áp dụng Hadamard:

$$
H \, |0⟩ = \frac{1}{\sqrt{2}} \, |0⟩ + \frac{1}{\sqrt{2}} \, |1⟩
$$

Trước khi đo:
✅ Hệ tồn tại đồng thời ở $|0⟩$ và $|1⟩$
✅ Khi đo:

* Xác suất đo $|0⟩$ là $\frac{1}{2}$
* Xác suất đo $|1⟩$ là $\frac{1}{2}$

---

# 🎯 **Bài tập Ngày 11 cho bạn**

1. Cho trạng thái ban đầu $|1⟩$
2. Áp dụng toán tử Hadamard lên $|1⟩$
3. Viết lại trạng thái sau biến đổi
4. Tính xác suất đo được $|0⟩$ và $|1⟩$

---

# 📝 **Giải mẫu Bài tập Ngày 11 - Toán tử Hadamard và Siêu vị**

---

## **Đề bài:**

Cho trạng thái ban đầu:

$$
|1⟩ = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

Áp dụng toán tử Hadamard:

$$
H = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
$$

Yêu cầu:

1. Tính trạng thái sau biến đổi
2. Viết rõ trạng thái mới theo $|0⟩$, $|1⟩$
3. Tính xác suất đo được $|0⟩$ và $|1⟩$

---

## ✅ **Bước 1: Tác động toán tử Hadamard lên $|1⟩$**

Tính:

$$
H \, |1⟩ = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \times \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

Viết lại theo $|0⟩$ và $|1⟩$:

$$
|\psi'⟩ = \frac{1}{\sqrt{2}} \, |0⟩ - \frac{1}{\sqrt{2}} \, |1⟩
$$

---

## ✅ **Bước 2: Kiểm tra chuẩn hóa**

Tính tổng bình phương hệ số:

$$
\left(\frac{1}{\sqrt{2}}\right)^2 + \left(-\frac{1}{\sqrt{2}}\right)^2 = \frac{1}{2} + \frac{1}{2} = 1
$$

✅ Chuẩn hóa hợp lệ.

---

## ✅ **Bước 3: Tính xác suất đo được $|0⟩$ và $|1⟩$**

* Xác suất đo được $|0⟩$:

$$
P(0) = \left( \frac{1}{\sqrt{2}} \right)^2 = \frac{1}{2} = 50\%
$$

* Xác suất đo được $|1⟩$:

$$
P(1) = \left( -\frac{1}{\sqrt{2}} \right)^2 = \frac{1}{2} = 50\%
$$

✅ Dù có dấu trừ, bình phương lên vẫn là dương, nên xác suất hợp lệ.

---

# 🎯 **Kết luận cuối cùng**

| Nội dung                | Kết quả |                             |                         |      |
| ----------------------- | ------- | --------------------------- | ----------------------- | ---- |
| Trạng thái sau biến đổi | (       | \psi'⟩ = \frac{1}{\sqrt{2}} | 0⟩ - \frac{1}{\sqrt{2}} | 1⟩ ) |
| Xác suất đo (           | 0⟩)     | 50%                         |                         |      |
| Xác suất đo (           | 1⟩)     | 50%                         |                         |      |
| Trạng thái chuẩn hóa    | Đúng    |                             |                         |      |

---

# 💡 **Ý nghĩa vật lý**

* Ban đầu hệ ở $|1⟩$ chắc chắn
* Sau khi áp dụng Hadamard, hệ trở thành siêu vị cân bằng: vừa $|0⟩$, vừa $|1⟩$
* Khi đo, kết quả có thể là $|0⟩$ hoặc $|1⟩$ với xác suất 50% mỗi bên
* Đây là cách máy tính lượng tử khai thác tính "song song" của hệ

