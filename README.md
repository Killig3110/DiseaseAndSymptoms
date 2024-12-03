# 22110118 Huỳnh Thanh Duy
# DiseaseAndSymptoms

Dự án **DiseaseAndSymptoms** ứng dụng thuật toán **CART (Classification and Regression Tree)** để dự đoán bệnh dựa trên các triệu chứng được cung cấp, đồng thời cung cấp các biện pháp phòng ngừa tương ứng. Dự án được triển khai trên bộ dữ liệu y tế, kết hợp học máy với xử lý dữ liệu để đạt hiệu quả cao trong dự đoán và phân tích.

## 🏥 Mục Tiêu Dự Án

- **Dự đoán bệnh:** Xác định bệnh dựa trên các triệu chứng được cung cấp.
- **Gợi ý phòng ngừa:** Cung cấp biện pháp phòng ngừa tương ứng với bệnh.
- **Phân tích hiệu suất:** Đánh giá hiệu suất mô hình qua các chỉ số và ma trận nhầm lẫn.
- **Tối ưu hóa mô hình:** Ứng dụng thuật toán CART với dữ liệu được chuẩn hóa và cân bằng.

## 📁 Cấu Trúc Dự Án

```
DiseaseAndSymptoms/
├── DiseaseAndSymptoms.csv       # Dữ liệu triệu chứng và bệnh
├── Disease precaution.csv       # Dữ liệu biện pháp phòng ngừa
├── classification_results.csv   # Kết quả phân loại
├── detailed_analysis_report.txt # Báo cáo chi tiết phân tích kết quả
├── symptom_mapping.csv          # Bảng mã hóa triệu chứng
├── disease_decision_tree_model.pkl # Mô hình cây quyết định
├── process_and_check_data.py    # Xử lý và kiểm tra dữ liệu
├── analyze_results.py           # Phân tích kết quả
├── Disease_and_Symptoms.py      # Huấn luyện và dự đoán
└── README.md                    # Mô tả dự án
```

## 🔧 Công Nghệ Sử Dụng

- Python
  - **pandas**: Xử lý dữ liệu.
  - **scikit-learn**: Triển khai thuật toán CART, mã hóa dữ liệu, và phân tích mô hình.
  - **imblearn (SMOTE)**: Cân bằng dữ liệu.
  - **matplotlib, seaborn**: Trực quan hóa dữ liệu.
- Jupyter Notebook (tùy chọn): Xây dựng và kiểm thử mô hình.
- Kaggle Dataset: Dữ liệu từ nguồn [Kaggle](https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset).

## 🚀 Hướng Dẫn Sử Dụng

### 1. Cài Đặt
```bash
# Clone repository
git clone https://github.com/Killig3110/DiseaseAndSymptoms.git
cd DiseaseAndSymptoms

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Chạy Dự Án
1. **Xử lý dữ liệu và kiểm tra:**
   ```bash
   python test_Disease.py
   ```

2. **Huấn luyện mô hình:**
   ```bash
   python Disease_and_Symptoms.py
   ```

3. **Phân tích kết quả:**
   ```bash
   python analyze_results.py
   ```

4. **Entropy và Giini Index**
   ```bash
   python symptom_ir_results.py
   ```

5. **Khôi phục dữ liệu và mapping dữ liệu**
   ```bash
   python process_and_check_data.py
   ```  

### 3. Output
- **classification_results.csv:** Kết quả dự đoán bệnh.
- **detailed_analysis_report.txt:** Báo cáo chi tiết phân tích kết quả.
- **Biểu đồ trực quan:** Các biểu đồ phân phối dữ liệu, tầm quan trọng của triệu chứng, ma trận nhầm lẫn.

## 📊 Phân Tích Kết Quả

### Độ Chính Xác
- Độ chính xác trên tập kiểm tra: `99.89%`
- Độ chính xác trên tập validation: `99.49%`

### Ma Trận Nhầm Lẫn
![ma_tran_nham_lan_test](https://github.com/user-attachments/assets/ad79a0b5-9b33-4c07-bef9-b92c8350f186)

### Tầm Quan Trọng Của Triệu Chứng
![tam_quan_trong_cua_trieu_chung_test](https://github.com/user-attachments/assets/43ab3ec1-feba-4eb2-bcbf-c529c1512455)

### Tỉ lệ dự đoán đúng theo từng bệnh
![ti_le_du_doan_dung](https://github.com/user-attachments/assets/593a0d1e-afb7-4f04-ab7f-747129f007e5)

## 📚 Tài Liệu Tham Khảo

1. Aurelien Geron, *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow*, O’Reilly, 2019.
2. GeeksforGeeks, ["CART (Classification and Regression Tree) in Machine Learning"](https://www.geeksforgeeks.org/cart-classification-and-regression-tree-in-machine-learning/), truy cập ngày 03 tháng 12 năm 2024.
3. Medium - GeekCulture, ["Decision Trees with CART Algorithm"](https://medium.com/geekculture/decision-trees-with-cart-algorithm-7e179acee8ff), truy cập ngày 03 tháng 12 năm 2024.
4. Kaggle Dataset, ["Disease and Symptoms Dataset"](https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset).

---

Bạn có thể chỉnh sửa nội dung này phù hợp với yêu cầu và phong cách của dự án. Nếu cần thêm hình ảnh hoặc chi tiết, hãy bổ sung tương ứng.
