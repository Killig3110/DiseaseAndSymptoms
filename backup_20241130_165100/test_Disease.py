import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import load
import matplotlib.pyplot as plt

# 1. Tải mô hình đã lưu
model = load("disease_decision_tree_model.pkl")  # Tải mô hình từ file đã lưu

# 2. Đọc dữ liệu huấn luyện (nếu cần tạo lại test_data.csv)
symptom_data = pd.read_csv("DiseaseAndSymptoms.csv")
symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]

# Mã hóa dữ liệu nếu cần
label_encoder = LabelEncoder()
for col in symptom_columns:
    symptom_data[col] = label_encoder.fit_transform(symptom_data[col].fillna(''))

# Tách dữ liệu và nhãn
X = symptom_data[symptom_columns]
y = label_encoder.fit_transform(symptom_data['Disease'])

# Chia lại tập dữ liệu (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Tạo file test_data.csv từ dữ liệu kiểm tra
test_data = X_test.copy()
test_data['Disease'] = label_encoder.inverse_transform(y_test)
test_data.to_csv("test_data.csv", index=False)
print("Dữ liệu kiểm tra đã được lưu vào file 'test_data.csv'.")

# 4. Đọc lại dữ liệu từ file test_data.csv
test_data = pd.read_csv("test_data.csv")

# 5. Chuẩn bị dữ liệu kiểm tra
X_new_test = test_data[symptom_columns]
y_new_test = test_data['Disease']

# 6. Mã hóa dữ liệu kiểm tra
for col in symptom_columns:
    X_new_test[col] = label_encoder.fit_transform(X_new_test[col])

y_new_test_encoded = label_encoder.fit_transform(y_new_test)

# 7. Dự đoán trên tập kiểm tra
y_new_pred = model.predict(X_new_test)

# 8. Đánh giá mô hình
print("Accuracy on Test Data:", accuracy_score(y_new_test_encoded, y_new_pred))
print("Classification Report:\n", classification_report(y_new_test_encoded, y_new_pred, target_names=label_encoder.classes_))

# 9. Vẽ ma trận nhầm lẫn
ConfusionMatrixDisplay.from_estimator(
    model, X_new_test, y_new_test_encoded, display_labels=label_encoder.classes_, cmap='viridis'
)
plt.title("Ma trận nhầm lẫn trên tập kiểm tra")
plt.show()

# Tạo bảng kết quả phân loại
results = X_new_test.copy()
results['True Disease'] = y_new_test  # Bệnh thực tế
results['Predicted Disease'] = label_encoder.inverse_transform(y_new_pred)  # Bệnh dự đoán

# Lưu bảng kết quả ra file CSV
results.to_csv("classification_results.csv", index=False)
print("Kết quả phân loại đã được lưu vào file 'classification_results.csv'.")
