Đồ án kết thúc môn Machine Learing:

Đề tài: Áp dụng thuật toán Decision Tree để giải quyết bài toán dự đoán bệnh dựa vào triệu chứng

Họ và tên: Huỳnh Thanh Duy

Mã số sinh viên: 22110118

Trường Đại học Sư phạm Kỹ thuật Thành phố Hồ Chí Minh

# Ứng dụng: **Dự đoán và Phân tích bệnh với thuật toán Cây Quyết định**

## **1. Tổng quan**
Ứng dụng bao gồm 3 giai đoạn:
1. **Tạo và huấn luyện mô hình**: Sử dụng `Disease_and_Symptoms.py` để xử lý dữ liệu và huấn luyện mô hình.
2. **Kiểm tra mô hình**: Chạy `test_Disease.py` để kiểm tra hiệu suất mô hình trên tập kiểm tra.
3. **Phân tích kết quả**: Dùng `analyze_results.py` để phân tích chi tiết và hiểu rõ hiệu quả của mô hình.

---

## **2. Cách sử dụng**
### **Bước 1: Tạo và Huấn luyện mô hình**
- **Mục đích**: Huấn luyện mô hình từ dữ liệu gốc.
- **File chạy**: `Disease_and_Symptoms.py`
- **Thao tác**:
  1. Đảm bảo các file dữ liệu sau có trong cùng thư mục:
     - `DiseaseAndSymptoms.csv`
     - `Disease precaution.csv`
  2. Chạy lệnh:
     ```bash
     python Disease_and_Symptoms.py
     ```
  3. Kết quả:
     - Mô hình huấn luyện được lưu thành `disease_decision_tree_model.pkl`.

---

### **Bước 2: Kiểm tra mô hình**
- **Mục đích**: Đánh giá hiệu suất mô hình trên tập kiểm tra.
- **File chạy**: `test_Disease.py`
- **Thao tác**:
  1. Đảm bảo file `disease_decision_tree_model.pkl` đã được tạo từ Bước 1.
  2. Chạy lệnh:
     ```bash
     python test_Disease.py
     ```
  3. Kết quả:
     - **File kiểm tra**: `test_data.csv` (Tập dữ liệu kiểm tra được lưu lại).
     - **Kết quả phân loại**: `classification_results.csv` (Kết quả dự đoán và biện pháp phòng ngừa).

---

### **Bước 3: Phân tích kết quả**
- **Mục đích**: Hiểu rõ hiệu suất của mô hình.
- **File chạy**: `analyze_results.py`
- **Thao tác**:
  1. Đảm bảo các file sau có trong thư mục:
     - `classification_results.csv`
     - `test_data.csv`
  2. Chạy lệnh:
     ```bash
     python analyze_results.py
     ```
  3. Kết quả:
     - **File phân tích chi tiết**: `detailed_analysis_report.txt` (Chứa các nhận xét cụ thể về kết quả).
     - **Hình ảnh phân tích**: Các biểu đồ trực quan.

---

## **3. Yêu cầu hệ thống**
- **Python**: >= 3.8
- **Thư viện cần thiết**:
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - joblib
  - imbalanced-learn (cài bằng `pip install imbalanced-learn`)

---

## **4. Cấu trúc thư mục**
```plaintext
.
├── Disease_and_Symptoms.py   # Huấn luyện mô hình
├── test_Disease.py           # Kiểm tra mô hình
├── analyze_results.py        # Phân tích kết quả
├── DiseaseAndSymptoms.csv    # Dữ liệu triệu chứng
├── Disease precaution.csv    # Biện pháp phòng ngừa
├── disease_decision_tree_model.pkl  # Mô hình đã huấn luyện
├── test_data.csv             # Tập kiểm tra
├── classification_results.csv # Kết quả phân loại
├── detailed_analysis_report.txt # Báo cáo phân tích
