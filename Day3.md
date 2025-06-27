
# 🌟 **Ngày 3: Ma Trận và Phép Nhân Ma Trận**

---

## **1. Ma trận là gì?**

✅ Ma trận là bảng gồm các số (hoặc biến) được sắp xếp theo hàng và cột.

**Ví dụ đơn giản:** Ma trận 2x2:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

**Ứng dụng:**

* Biểu diễn phép biến đổi tuyến tính trong không gian
* Trong lượng tử, ma trận được dùng để mô tả các toán tử tác động lên trạng thái lượng tử

---

## **2. Phép Nhân Ma Trận**

Muốn nhân hai ma trận:

* Số cột của ma trận thứ nhất phải bằng số hàng của ma trận thứ hai
* Kết quả là ma trận có số hàng của ma trận đầu và số cột của ma trận sau

### **Công thức:**

Cho hai ma trận:

$$
A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}, \quad B = \begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{bmatrix}
$$

Tích $C = A \times B$:

$$
C_{ij} = \sum_k a_{ik} \times b_{kj}
$$

---

## **3. Ví dụ cụ thể**

Cho:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

Tính:

$$
C = A \times B = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

Từng phần tử:

* $C_{11} = (1 \times 0) + (2 \times 1) = 2$
* $C_{12} = (1 \times 1) + (2 \times 0) = 1$
* $C_{21} = (3 \times 0) + (4 \times 1) = 4$
* $C_{22} = (3 \times 1) + (4 \times 0) = 3$

Vậy kết quả:

$$
C = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}
$$

---

## **4. Ý nghĩa trong lượng tử**

* Trạng thái lượng tử được viết dạng vector cột
* Các toán tử lượng tử được biểu diễn bởi ma trận
* Phép nhân ma trận chính là cách toán tử tác động lên trạng thái lượng tử

---

# 🎯 **Bài tập Ngày 3 cho bạn**

Cho:

$$
A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}, \quad B = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
$$

Tính tích $C = A \times B$.

