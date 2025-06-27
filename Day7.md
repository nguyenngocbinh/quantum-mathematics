
# 🧩 **Ngày 7: Trạng thái lượng tử và Hàm sóng**

---

## **1. Trạng thái lượng tử là gì?**

* Trong toán lượng tử, trạng thái của một hệ (hạt, photon, electron…) được biểu diễn bằng **vector** trong **không gian Hilbert**.
* Vector trạng thái thường ký hiệu là:

$$
|\psi⟩
$$

Gọi là **ket**, theo ký hiệu Dirac.

---

## **2. Dạng tổng quát của trạng thái lượng tử**

Với hệ đơn giản như **1 qubit** (hệ nhị phân, giống như 0 và 1):

$$
|\psi⟩ = a|0⟩ + b|1⟩
$$

Trong đó:

✅ $|0⟩ = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
✅ $|1⟩ = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$
✅ **a, b** là số phức hoặc số thực, thỏa mãn:

$$
|a|^2 + |b|^2 = 1
$$

Điều kiện này đảm bảo tổng xác suất đo được là 100%.

---

## **3. Hàm sóng (Wave Function) là gì?**

* Hàm sóng là cách mô tả trạng thái lượng tử trong ngôn ngữ toán học.
* Với hệ liên tục (như vị trí hạt trên trục x), hàm sóng là:

$$
\psi(x)
$$

Trong đó:
✅ $|\psi(x)|^2$ cho biết xác suất tìm thấy hạt tại vị trí **x**
✅ Hàm sóng phải được chuẩn hóa:

$$
\int_{-\infty}^{\infty} |\psi(x)|^2 \, dx = 1
$$

**Lưu ý:** Với hệ rời rạc (như qubit $|0⟩, |1⟩$), ta chỉ cần tổng bình phương các hệ số bằng 1.

---

## **4. Ví dụ minh họa:**

Giả sử trạng thái:

$$
|\psi⟩ = \frac{1}{\sqrt{2}} |0⟩ + \frac{1}{\sqrt{2}} |1⟩
$$

Tính:

* $|a|^2 + |b|^2 = \left(\frac{1}{\sqrt{2}}\right)^2 + \left(\frac{1}{\sqrt{2}}\right)^2 = \frac{1}{2} + \frac{1}{2} = 1$
* Xác suất đo được:

  * $|0⟩$: 50%
  * $|1⟩$: 50%

**Đây là trạng thái siêu vị tiêu chuẩn (Superposition state)**, bạn sẽ gặp rất nhiều sau này.

---

# 🎯 **Bài tập Ngày 7 cho bạn:**

Viết 2 trạng thái lượng tử khác nhau, đảm bảo chuẩn hóa:

**Ví dụ:**
Trạng thái 1: $|\psi_1⟩ = a|0⟩ + b|1⟩$ với $a = \sqrt{0.7}, b = \sqrt{0.3}$

Trạng thái 2: Bạn tự chọn hệ số khác, thỏa mãn $|a|^2 + |b|^2 = 1$

---


# 📝 **Giải mẫu bài tập Ngày 7 - Trạng thái lượng tử chuẩn hóa**

---

## **Trạng thái 1:**

Đề bài gợi ý:

$$
|\psi_1⟩ = \sqrt{0.7} \, |0⟩ + \sqrt{0.3} \, |1⟩
$$

**Kiểm tra điều kiện chuẩn hóa:**

$$
|a|^2 + |b|^2 = (\sqrt{0.7})^2 + (\sqrt{0.3})^2 = 0.7 + 0.3 = 1
$$

✅ Chuẩn hóa hợp lệ.

**Xác suất đo được:**

* Trạng thái $|0⟩$: 70%
* Trạng thái $|1⟩$: 30%

---

## **Trạng thái 2:**

Tôi tự chọn một trạng thái hợp lệ:

$$
|\psi_2⟩ = 0.6 \, |0⟩ + 0.8 \, |1⟩
$$

**Kiểm tra điều kiện chuẩn hóa:**

$$
|a|^2 + |b|^2 = (0.6)^2 + (0.8)^2 = 0.36 + 0.64 = 1
$$

✅ Chuẩn hóa hợp lệ.

**Xác suất đo được:**

* Trạng thái $|0⟩$: 36%
* Trạng thái $|1⟩$: 64%

---

# 🎯 **Tóm tắt bạn cần nhớ:**

* Trạng thái lượng tử luôn viết dưới dạng:

$$
|\psi⟩ = a \, |0⟩ + b \, |1⟩
$$

* Phải thỏa mãn $|a|^2 + |b|^2 = 1$
* Xác suất đo được chính là bình phương hệ số tương ứng

---

