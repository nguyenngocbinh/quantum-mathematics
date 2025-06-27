
# 🌟 **Ngày 15: Cổng lượng tử nhiều qubit — Tập trung vào Cổng CNOT**

---

## **1. Cổng lượng tử là gì?**

* Cổng lượng tử giống như các phép toán trong máy tính cổ điển
* Tuy nhiên, cổng lượng tử tác động lên trạng thái lượng tử (qubit)
* Có cổng đơn qubit (Hadamard, Pauli-X...)
* Có cổng hai qubit trở lên — quan trọng nhất là CNOT

---

## **2. Cổng CNOT — Controlled-NOT**

✅ Là cổng điều kiện giữa hai qubit:

* Một qubit điều khiển (Control qubit)
* Một qubit đích (Target qubit)

**Nguyên lý hoạt động:**

* Nếu Control = $|0⟩$ ⇒ không làm gì
* Nếu Control = $|1⟩$ ⇒ áp dụng Pauli-X (lật trạng thái) cho qubit đích

**Ma trận CNOT (4x4):**

$$
\text{CNOT} = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0
\end{bmatrix}
$$

Cách hiểu: thứ tự các trạng thái:

$$
|00⟩, \, |01⟩, \, |10⟩, \, |11⟩
$$

---

## **3. Ví dụ cụ thể**

Giả sử:

* Trạng thái 2 qubit:

$$
|\Psi⟩ = \frac{1}{\sqrt{2}} \, |00⟩ + \frac{1}{\sqrt{2}} \, |10⟩
$$

* Áp dụng CNOT:

  * Khi Control qubit là $|0⟩$ → không làm gì
  * Khi Control qubit là $|1⟩$ → lật qubit thứ hai

**Từng bước:**

* Thành phần $|00⟩$ → Control là $|0⟩$ ⇒ không đổi
* Thành phần $|10⟩$ → Control là $|1⟩$ ⇒ qubit thứ hai lật từ $|0⟩$ thành $|1⟩$ ⇒ $|11⟩$

Vậy:

$$
\text{CNOT} \, |\Psi⟩ = \frac{1}{\sqrt{2}} \, |00⟩ + \frac{1}{\sqrt{2}} \, |11⟩
$$

Đây chính là trạng thái rối lượng tử nổi tiếng $|\Phi^+⟩$.

---

## **4. Tóm tắt: Cổng CNOT rất quan trọng vì**

✅ Cho phép điều khiển tương tác giữa các qubit
✅ Là công cụ tạo ra rối lượng tử
✅ Cơ sở để xây dựng thuật toán lượng tử phức tạp

---

# 🎯 **Bài tập Ngày 15 cho bạn**

Cho:

* Trạng thái ban đầu:

$$
|\Psi⟩ = \frac{1}{\sqrt{2}} \, |00⟩ + \frac{1}{\sqrt{2}} \, |10⟩
$$

* Áp dụng Cổng CNOT (qubit đầu tiên là Control, qubit thứ hai là Target)

Yêu cầu:

1. Viết rõ trạng thái sau khi áp dụng CNOT
2. Kiểm tra trạng thái sau có phải trạng thái rối lượng tử không?


---

# 📝 **Giải mẫu Bài tập Ngày 15 — Cổng CNOT và Trạng thái rối lượng tử**

---

## **Đề bài:**

Cho:

* Trạng thái ban đầu:

$$
|\Psi⟩ = \frac{1}{\sqrt{2}} \, |00⟩ + \frac{1}{\sqrt{2}} \, |10⟩
$$

* Áp dụng Cổng CNOT:

  * Qubit thứ nhất là Control
  * Qubit thứ hai là Target

Yêu cầu:

1. Tính trạng thái sau khi áp dụng CNOT
2. Kiểm tra trạng thái sau có phải rối lượng tử không?

---

# ✅ **Bước 1: Áp dụng Cổng CNOT**

Nhắc lại:

* Nếu Control = $|0⟩$ ⇒ không làm gì
* Nếu Control = $|1⟩$ ⇒ lật Target qubit

**Từng thành phần:**

* Thành phần $|00⟩$:

  * Control là $|0⟩$
  * Không thay đổi ⇒ Giữ nguyên $|00⟩$

* Thành phần $|10⟩$:

  * Control là $|1⟩$
  * Target lật từ $|0⟩$ thành $|1⟩$ ⇒ $|11⟩$

Vậy:

$$
\text{CNOT} \, |\Psi⟩ = \frac{1}{\sqrt{2}} \, |00⟩ + \frac{1}{\sqrt{2}} \, |11⟩
$$

Đây chính là trạng thái:

$$
|\Phi^+⟩ = \frac{1}{\sqrt{2}} \, \left( |00⟩ + |11⟩ \right)
$$

---

# ✅ **Bước 2: Kiểm tra trạng thái rối lượng tử**

* Trạng thái $|\Phi^+⟩$ là trạng thái nổi tiếng thuộc họ Bell States
* Không thể tách trạng thái này thành tích của hai qubit riêng lẻ
* Đây là trạng thái rối lượng tử mạnh, thể hiện sự liên kết giữa hai qubit

✅ Kết luận: Trạng thái sau khi áp dụng CNOT là trạng thái rối lượng tử.

---

# 🎯 **Tóm tắt kết quả**

| Câu hỏi                                    | Đáp án                                               |                            |       |
| ------------------------------------------ | ---------------------------------------------------- | -------------------------- | ----- |
| Trạng thái sau CNOT                        | ( \frac{1}{\sqrt{2}} ,                               | 00⟩ + \frac{1}{\sqrt{2}} , | 11⟩ ) |
| Trạng thái này có phải rối lượng tử không? | ✅ Có, thuộc họ Bell States, không thể tách rời qubit |                            |       |

---

# 💡 **Ý nghĩa vật lý**

* Cổng CNOT là công cụ đơn giản nhưng cực kỳ mạnh mẽ để tạo ra rối lượng tử
* Máy tính lượng tử khai thác sự rối để xử lý thông tin theo cách máy cổ điển không làm được
* Bạn vừa thực hành nguyên lý cơ bản của lập trình lượng tử thực tế

