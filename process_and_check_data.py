import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Đọc dữ liệu
symptom_data = pd.read_csv("DiseaseAndSymptoms.csv")
symptom_data = symptom_data.fillna('')  # Điền giá trị rỗng nếu có

# 2. Xác định cột triệu chứng và bệnh
symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]
X = symptom_data[symptom_columns]
y = symptom_data['Disease']

# 3. Mã hóa từng cột triệu chứng và lưu lại encoder
encoders = {}  # Lưu trữ LabelEncoder cho từng cột
for col in symptom_columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])  # Mã hóa cột triệu chứng
    encoders[col] = encoder  # Lưu LabelEncoder của cột

# Mã hóa nhãn bệnh
disease_encoder = LabelEncoder()
y_encoded = disease_encoder.fit_transform(y)

# 4. Lưu các LabelEncoder và dữ liệu mã hóa
joblib.dump(encoders, "symptom_encoders.pkl")  # Lưu các encoder của triệu chứng
joblib.dump(disease_encoder, "disease_encoder.pkl")  # Lưu encoder của bệnh

# Lưu dữ liệu đã mã hóa
encoded_data = pd.concat([X, pd.Series(y_encoded, name='Disease')], axis=1)
encoded_data.to_csv("encoded_data.csv", index=False)

print("Dữ liệu mã hóa và các encoder đã được lưu.")

# 5. Tạo Mapping triệu chứng (cho từng cột triệu chứng riêng)
symptom_mapping_list = []
for col in symptom_columns:
    symptom_mapping = dict(zip(encoders[col].classes_, range(len(encoders[col].classes_))))
    for symptom, encoded_value in symptom_mapping.items():
        symptom_mapping_list.append([col, symptom, encoded_value])

symptom_mapping_df = pd.DataFrame(symptom_mapping_list, columns=["Symptom Column", "Symptom", "Encoded Value"])
symptom_mapping_df.to_csv("symptom_mapping.csv", index=False)

# Tạo Mapping bệnh (tên bệnh -> giá trị mã hóa)
disease_mapping = dict(zip(disease_encoder.classes_, range(len(disease_encoder.classes_))))

# Tạo DataFrame cho mapping bệnh
disease_mapping_df = pd.DataFrame(disease_mapping.items(), columns=["Disease", "Encoded Value"])
disease_mapping_df.to_csv("disease_mapping.csv", index=False)

print("Mapping triệu chứng và bệnh đã được lưu.")

# 6. Phục hồi dữ liệu gốc từ mã hóa
# Đọc dữ liệu mã hóa và các LabelEncoder
restored_data = pd.read_csv("encoded_data.csv")
encoders = joblib.load("symptom_encoders.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")

# Phục hồi từng cột triệu chứng
for col in symptom_columns:
    encoder = encoders[col]  # Lấy encoder của cột
    restored_data[col] = encoder.inverse_transform(restored_data[col])

# Phục hồi nhãn bệnh
restored_data['Disease'] = disease_encoder.inverse_transform(restored_data['Disease'])

# Lưu dữ liệu đã khôi phục
restored_data.to_csv("restored_data_checked.csv", index=False)
print("Dữ liệu đã được khôi phục và lưu vào 'restored_data_checked.csv'.")
