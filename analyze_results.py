import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# 1. Đọc file kết quả và mapping
symptom_mapping = pd.read_csv("symptom_mapping.csv")
results = pd.read_csv("classification_results.csv")
disease_data = pd.read_csv("DiseaseAndSymptoms.csv")

# Tạo dictionary ánh xạ cho từng cột triệu chứng
column_symptom_dict = {
    col: symptom_mapping[symptom_mapping['Symptom Column'] == col]
    .set_index('Encoded Value')['Symptom']
    .to_dict()
    for col in symptom_mapping['Symptom Column'].unique()
}

symptom_columns = [col for col in disease_data.columns if col.startswith('Symptom')]

# 2. Tổng quan dữ liệu
print("Thông tin dữ liệu:")
print(results.info())
print("\nMẫu dữ liệu:")
print(results.head())

# 3. Đánh giá độ chính xác tổng thể
correct_predictions = (results['True Disease'] == results['Predicted Disease']).sum()
total_predictions = len(results)
accuracy = correct_predictions / total_predictions

print(f"\nSố dự đoán đúng: {correct_predictions}")
print(f"Tổng số dự đoán: {total_predictions}")
print(f"Độ chính xác tổng thể: {accuracy:.2%}")

# 4. Vẽ ma trận nhầm lẫn
print("\nVẽ ma trận nhầm lẫn:")
label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(results['True Disease'])
y_pred = label_encoder.transform(results['Predicted Disease'])

fig, ax = plt.subplots(figsize=(20, 20))
disp = ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred, display_labels=label_encoder.classes_, cmap='viridis', ax=ax
)
plt.title("Ma trận nhầm lẫn")
plt.xticks(rotation=90, fontsize=10, ha='right')
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# 5. Phân tích lỗi
print("\nPhân tích lỗi:")
wrong_predictions = results[results['True Disease'] != results['Predicted Disease']]
wrong_by_true = wrong_predictions['True Disease'].value_counts()
wrong_by_pred = wrong_predictions['Predicted Disease'].value_counts()

print("Các bệnh thường bị dự đoán sai:")
print(wrong_by_true.head(5))

print("\nCác bệnh thường bị dự đoán sai thành:")
print(wrong_by_pred.head(5))

# 6. Nguyên nhân dựa trên tiêu chí đánh giá
print("\nPhân tích nguyên nhân dựa trên kết quả:")

# Lấy triệu chứng cho các dự đoán sai
wrong_symptoms = wrong_predictions[symptom_columns].copy()

# Thay thế mã hóa bằng tên gốc
for col in symptom_columns:
    if col in column_symptom_dict:
        wrong_symptoms[col] = wrong_symptoms[col].map(column_symptom_dict[col])

# Gộp tất cả triệu chứng để phân tích
common_wrong_symptoms = (
    wrong_symptoms.apply(pd.Series.value_counts)
    .sum(axis=1)
    .sort_values(ascending=False)
)

print("Triệu chứng phổ biến trong các dự đoán sai (tên gốc):")
print(common_wrong_symptoms.head(10))

# Phân phối dữ liệu không đồng đều
true_disease_distribution = results['True Disease'].value_counts()
print("\nPhân phối số mẫu dữ liệu theo bệnh:")
print(true_disease_distribution)

# Xác định nguyên nhân từ dữ liệu
causes = []

# Nguyên nhân: Số lượng mẫu không đồng đều
if true_disease_distribution.std() > 0.1 * true_disease_distribution.mean():
    causes.append(
        "Dữ liệu không đồng đều: Một số bệnh có quá ít mẫu dữ liệu so với các bệnh khác."
    )

# Nguyên nhân: Triệu chứng dễ gây nhầm lẫn
if not common_wrong_symptoms.empty:
    top_wrong_symptoms = common_wrong_symptoms.head(5).index.tolist()
    causes.append(
        f"Triệu chứng dễ gây nhầm lẫn: Các triệu chứng phổ biến trong dự đoán sai bao gồm {', '.join(map(str, top_wrong_symptoms))}."
    )

# Nguyên nhân: Tỷ lệ dự đoán sai cao với một số bệnh
if not wrong_by_true.empty:
    top_wrong_diseases = wrong_by_true.head(3).index.tolist()
    causes.append(
        f"Các bệnh thường bị dự đoán sai: Bao gồm {', '.join(top_wrong_diseases)}."
    )

# Hiển thị các nguyên nhân
if causes:
    print("\nNguyên nhân tiềm ẩn:")
    for i, cause in enumerate(causes, start=1):
        print(f"{i}. {cause}")
else:
    print("\nKhông tìm thấy nguyên nhân rõ ràng.")

# 7. Tỷ lệ dự đoán đúng theo từng bệnh
print("\nTỷ lệ dự đoán đúng theo từng bệnh:")
accuracy_by_disease = results.groupby('True Disease').apply(
    lambda group: (group['True Disease'] == group['Predicted Disease']).mean()
)

print(accuracy_by_disease)

# Vẽ biểu đồ tỷ lệ dự đoán đúng
plt.figure(figsize=(10, 8))
accuracy_by_disease.sort_values().plot(kind='barh', color='skyblue')
plt.title("Tỷ lệ dự đoán đúng theo từng bệnh")
plt.xlabel("Độ chính xác")
plt.ylabel("Bệnh")
plt.show()

# 8. Lưu báo cáo vào file
report_file = "detailed_analysis_report.txt"
with open(report_file, "w", encoding="utf-8") as f:
    f.write("Báo cáo Phân Tích Kết Quả\n")
    f.write(f"Tổng số dự đoán: {total_predictions}\n")
    f.write(f"Số dự đoán đúng: {correct_predictions}\n")
    f.write(f"Độ chính xác tổng thể: {accuracy:.2%}\n\n")

    f.write("Tỷ lệ dự đoán đúng theo từng bệnh:\n")
    f.write(accuracy_by_disease.to_string())
    f.write("\n\nCác bệnh thường bị dự đoán sai:\n")
    f.write(wrong_by_true.to_string())
    f.write("\n\nCác bệnh thường bị dự đoán sai thành:\n")
    f.write(wrong_by_pred.to_string())
    f.write("\n\nTriệu chứng phổ biến trong các dự đoán sai (tên gốc):\n")
    f.write(common_wrong_symptoms.to_string())
    f.write("\n\nPhân phối số mẫu dữ liệu theo bệnh:\n")
    f.write(true_disease_distribution.to_string())
    f.write("\n\nNguyên nhân tiềm ẩn:\n")
    for cause in causes:
        f.write(f"- {cause}\n")
print(f"\nBáo cáo phân tích chi tiết đã được lưu vào file '{report_file}'.")
