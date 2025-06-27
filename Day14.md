
# 🌟 **Ngày 14: Toán học của Hệ Nhiều Qubit và Trạng thái Tổng hợp**

---

## **1. Hệ nhiều qubit là gì?**

✅ Một qubit đơn có thể ở trạng thái:

$$
|\psi\rangle = a \, |0\rangle + b \, |1\rangle
$$

Với điều kiện chuẩn hóa:

$$
|a|^2 + |b|^2 = 1
$$

✅ Khi kết hợp nhiều qubit:

* Số trạng thái khả thi tăng theo lũy thừa
* Hệ có $n$ qubit ⇒ tồn tại $2^n$ trạng thái cơ bản

---

## **2. Tích tensor — Công cụ kết hợp qubit**

Tích tensor (ký hiệu $\otimes$) giúp xây dựng trạng thái tổng hợp:

**Ví dụ:**

* Qubit 1: $|\psi_1\rangle = a_1 \, |0\rangle + b_1 \, |1\rangle$
* Qubit 2: $|\psi_2\rangle = a_2 \, |0\rangle + b_2 \, |1\rangle$

Trạng thái hệ 2 qubit:

$$
|\Psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle = a_1 a_2 |00\rangle + a_1 b_2 |01\rangle + b_1 a_2 |10\rangle + b_1 b_2 |11\rangle
$$

---

## **3. Không gian Hilbert của nhiều qubit**

* Hệ $n$ qubit có không gian Hilbert kích thước $2^n$
* Số trạng thái cơ bản là:

$$
|00\ldots 0\rangle, \quad |00\ldots 1\rangle, \quad \ldots, \quad |11\ldots 1\rangle
$$

✅ Đây là lý do máy tính lượng tử có thể xử lý song song cực nhiều khả năng cùng lúc

---

## **4. Ví dụ cụ thể - 2 Qubit**

Cho:

* Qubit 1: $|\psi_1\rangle = \sqrt{0.6} \, |0\rangle + \sqrt{0.4} \, |1\rangle$
* Qubit 2: $|\psi_2\rangle = |1\rangle$

Tính trạng thái tổng hợp:

$$
|\Psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle = \sqrt{0.6} \, |0\rangle \otimes |1\rangle + \sqrt{0.4} \, |1\rangle \otimes |1\rangle = \sqrt{0.6} \, |01\rangle + \sqrt{0.4} \, |11\rangle
$$

---

## **5. Trạng thái rối hay không rối**

* Nếu trạng thái tổng hợp có thể viết dạng tích tensor của từng qubit riêng ⇒ không rối
* Nếu không thể tách rời ⇒ trạng thái rối (entangled)

**Ví dụ:**

Trạng thái:

$$
|\Phi^+\rangle = \frac{1}{\sqrt{2}} ( |00\rangle + |11\rangle )
$$

Không thể tách rời thành tích của từng qubit riêng biệt ⇒ Đây là trạng thái rối lượng tử.

---

# 🎯 **Bài tập Ngày 14 cho bạn**

1. Cho: $|\psi_1\rangle = |0\rangle$, $|\psi_2\rangle = \sqrt{0.5} \, |0\rangle + \sqrt{0.5} \, |1\rangle$
   Tính trạng thái tổng hợp $|\Psi\rangle$

2. Trạng thái tổng hợp trên có phải trạng thái rối lượng tử không? Giải thích ngắn gọn.

---


# 📝 **Giải mẫu Bài tập Ngày 14 — Hệ nhiều qubit và Trạng thái tổng hợp**

---

## **Đề bài:**

Cho:

* Qubit 1: $|\psi_1\rangle = |0\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
* Qubit 2: $|\psi_2\rangle = \sqrt{0.5} \, |0\rangle + \sqrt{0.5} \, |1\rangle = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$

Yêu cầu:

1. Tính trạng thái tổng hợp $|\Psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle$
2. Kiểm tra trạng thái tổng hợp có phải là trạng thái rối lượng tử không?

---

# ✅ **Bước 1: Tính trạng thái tổng hợp**

Tích tensor:

$$
|\Psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle = |0\rangle \otimes \left( \frac{1}{\sqrt{2}} \, |0\rangle + \frac{1}{\sqrt{2}} \, |1\rangle \right)
$$

Phân phối:

$$
|\Psi\rangle = \frac{1}{\sqrt{2}} \, |0\rangle \otimes |0\rangle + \frac{1}{\sqrt{2}} \, |0\rangle \otimes |1\rangle = \frac{1}{\sqrt{2}} \, |00\rangle + \frac{1}{\sqrt{2}} \, |01\rangle
$$

---

# ✅ **Bước 2: Kiểm tra có phải trạng thái rối không?**

**Nhận xét:**

* Trạng thái tổng hợp là:

$$
|\Psi\rangle = \frac{1}{\sqrt{2}} \, |00\rangle + \frac{1}{\sqrt{2}} \, |01\rangle
$$

* Đây là kết quả của việc lấy tích tensor của từng qubit riêng lẻ, ta đã tính cụ thể từ đầu:

$$
|\Psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle
$$

✅ Khi trạng thái tổng hợp viết được dưới dạng tích riêng của từng qubit ⇒ **Không phải trạng thái rối lượng tử**

---

# 🎯 **Tóm tắt kết quả**

| Câu hỏi                              | Đáp án                                              |                        |                                  |             |
| ------------------------------------ | --------------------------------------------------- | ---------------------- | -------------------------------- | ----------- |
| Trạng thái tổng hợp (                | \Psi\rangle )                                       | ( \frac{1}{\sqrt{2}} , | 00\rangle + \frac{1}{\sqrt{2}} , | 01\rangle ) |
| Đây có phải trạng thái rối lượng tử? | ❌ Không, vì có thể tách riêng thành tích từng qubit |                        |                                  |             |

---

# 💡 **Lưu ý quan trọng**

* Không phải cứ hệ nhiều qubit là trạng thái rối
* Chỉ khi không thể tách riêng thành tích các qubit riêng lẻ, mới gọi là rối lượng tử
* Rối lượng tử thể hiện sự phụ thuộc chặt chẽ giữa các qubit, cần thiết cho nhiều ứng dụng lượng tử tiên tiến
