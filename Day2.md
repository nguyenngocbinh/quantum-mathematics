
# 🧩 **Ngày 2: Tích vô hướng (Dot Product) và Ý nghĩa trong hình học**

---

## **1. Tích vô hướng là gì?**

**Tích vô hướng** là phép toán giữa 2 vector cho ra một số thực (không phải vector mới).

**Công thức:**

Cho hai vector:

$$
\mathbf{A} = (x_1, y_1), \quad \mathbf{B} = (x_2, y_2)
$$

Tích vô hướng của A và B là:

$$
\mathbf{A} \cdot \mathbf{B} = x_1 \times x_2 + y_1 \times y_2
$$

---

## **2. Ý nghĩa hình học**

Tích vô hướng liên quan đến góc giữa hai vector:

$$
\mathbf{A} \cdot \mathbf{B} = |\mathbf{A}| \times |\mathbf{B}| \times \cos(\theta)
$$

* **|A|** và **|B|** là độ lớn (độ dài) của hai vector
* **θ** là góc giữa chúng
* Nếu:

  * **Tích vô hướng > 0** → Góc < 90°, hai vector "cùng hướng" phần nào
  * **Tích vô hướng < 0** → Góc > 90°, hai vector "ngược hướng" phần nào
  * **Tích vô hướng = 0** → Hai vector vuông góc

---

## **3. Ví dụ tính tích vô hướng**

Cho:

$$
\mathbf{A} = (3, 4), \quad \mathbf{B} = (1, 2)
$$

Tính:

$$
\mathbf{A} \cdot \mathbf{B} = 3 \times 1 + 4 \times 2 = 3 + 8 = 11
$$

Tiếp theo, tính độ lớn từng vector:

* **|A|** = √(3² + 4²) = 5
* **|B|** = √(1² + 2²) = √5 ≈ 2.236

Tìm góc giữa A và B:

$$
\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{A}| \times |\mathbf{B}|} = \frac{11}{5 \times 2.236} ≈ 0.9839
$$

Vậy:

$$
\theta ≈ \cos^{-1}(0.9839) ≈ 10^\circ
$$

➡️ Hai vector gần như cùng hướng, góc rất nhỏ.

---

# 🎯 **Bài tập Ngày 2 cho bạn:**

Cho:

$$
\mathbf{C} = (5, -2), \quad \mathbf{D} = (-3, 7)
$$

Tính:

1. **C · D**
2. Độ lớn của C và D
3. Góc giữa C và D (bằng công thức trên)

---

## **Ghi chú nhỏ cho lượng tử:**

Trong toán lượng tử:

* Trạng thái được biểu diễn bằng vector
* Tích vô hướng giữa 2 trạng thái cho biết **sự tương đồng** hoặc **xác suất chuyển trạng thái**

---

