# 🔷 E(3)-Equivariant Neural Networks – Thực nghiệm và Phân tích

## 📌 Giới thiệu đề tài

Đề tài này tập trung nghiên cứu và đánh giá **E(3)-Equivariant Neural Networks (E3NN)** – một lớp mô hình học sâu tích hợp trực tiếp các **đối xứng hình học trong không gian 3D** (tịnh tiến, quay và phản xạ) vào kiến trúc mạng nơ-ron.

Thông qua ba kịch bản thực nghiệm, đề tài làm rõ vai trò của **tính đẳng biến hình học (equivariance)** đối với:
- Khả năng **tổng quát hóa dưới phép quay**
- **Hiệu quả sử dụng dữ liệu** trong bối cảnh dữ liệu hạn chế
- **Hiệu năng trên bài toán thực tế quy mô lớn** trong hóa học lượng tử (QM9)

🎯 Mục tiêu cuối cùng là trả lời câu hỏi:  
> *Việc tích hợp inductive bias hình học có thực sự mang lại lợi ích thực tiễn so với các mô hình học sâu truyền thống hay không?*

---

## 🧪 Các kịch bản thực nghiệm

| Kịch bản | Nội dung | Mục tiêu |
|--------|---------|----------|
| 🧪 **Thí nghiệm 1** | Kiểm chứng tính đẳng biến theo phép quay $SO(3)$ | Đánh giá khả năng bảo toàn cấu trúc hình học |
| 📉 **Thí nghiệm 2** | Hiệu quả sử dụng dữ liệu (Data Efficiency) | So sánh MAE khi số mẫu huấn luyện thay đổi |
| ⚛️ **Thí nghiệm 3** | Dự đoán năng lượng phân tử QM9 | Đánh giá trên bài toán thực tế quy mô lớn |

---

## 🛠️ Cài đặt môi trường

### 1️⃣ Yêu cầu hệ thống
- Python ≥ **3.9**
- Khuyến nghị: **GPU (CUDA)** để chạy nhanh hơn

### 2️⃣ Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv e3nn_env
source e3nn_env/bin/activate   # Linux / MacOS
# e3nn_env\Scripts\activate    # Windows
```
### 3️⃣ Cài đặt các thư viện cần thiết
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install e3nn
pip install torch-geometric
pip install numpy matplotlib tqdm scikit-learn
```

### ▶️ Hướng dẫn chạy thí nghiệm

Mỗi kịch bản thực nghiệm được cài đặt trong một file Python riêng biệt.

### 🧪 Thí nghiệm 1: Kiểm chứng tính đẳng biến
```bash
python test_1.py
```

📌 Kết quả:

So sánh accuracy giữa Baseline và E3NN

Đánh giá hiệu năng khi dữ liệu bị xoay ngẫu nhiên trong $SO(3)$

### 📉 Thí nghiệm 2: Hiệu quả sử dụng dữ liệu
```bash
python test_2.py
```

📌 Kết quả:

MAE trên tập test với các kích thước tập huấn luyện khác nhau

Sinh biểu đồ log–log thể hiện data efficiency

### ⚛️ Thí nghiệm 3: Dự đoán năng lượng phân tử QM9
```bash
python test_3.py
```

### 📌 Kết quả:

So sánh MAE giữa Invariant GNN và E3NN trên dataset QM9

Sinh biểu đồ quá trình hội tụ trong quá trình huấn luyện

⏳ Lưu ý: Lần chạy đầu tiên sẽ tự động tải dataset QM9 (~300MB).

