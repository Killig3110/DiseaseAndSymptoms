import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
from sklearn.utils import shuffle

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# 1. Đọc dữ liệu
symptom_data = pd.read_csv("DiseaseAndSymptoms.csv")
precaution_data = pd.read_csv("Disease precaution.csv")

# 2. Trộn dữ liệu để đảm bảo ngẫu nhiên
symptom_data = shuffle(symptom_data, random_state=42) # Trộn dữ liệu để đảm bảo ngẫu nhiên và phân phối đồng đều các bệnh

# 2. Xử lý dữ liệu triệu chứng
symptom_data = symptom_data.fillna('')
symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]
X = symptom_data[symptom_columns]
y = symptom_data['Disease']

# 3. Mã hóa dữ liệu
label_encoder = LabelEncoder()
for col in symptom_columns:
    X.loc[:, col] = label_encoder.fit_transform(X[col])

y = label_encoder.fit_transform(y)

# 4. Cân bằng dữ liệu bằng SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 5. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 6. Huấn luyện mô hình với các tham số tối ưu
model = DecisionTreeClassifier(
    random_state=42,
    criterion='gini',
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2
)
model.fit(X_train, y_train)

import pandas as pd
import matplotlib.pyplot as plt

# Vẽ biểu đồ đường sau khi làm sạch dữ liệu
def classification_report_to_csv_and_plot(report, output_path="classification_report.csv"):
    # Chuyển classification report thành dataframe
    report_data = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Loại bỏ các giá trị không liên quan
            metrics['label'] = label
            report_data.append(metrics)
    
    df = pd.DataFrame(report_data)
    
    # Xóa cột không cần thiết
    if "" in df.columns:
        df = df.drop(columns=[""])  # Xóa cột "" nếu tồn tại
    
    # Lưu kết quả ra file CSV
    df.to_csv(output_path, index=False)
    print(f"Classification Report đã được lưu vào {output_path}")
    
    # Vẽ biểu đồ đường
    df = df[df['label'] != 'accuracy']  # Loại bỏ hàng tổng kết nếu có
    labels = df['label']
    plt.figure(figsize=(20, 8))  # Tăng kích thước để dễ hiển thị hơn
    
    plt.plot(labels, df['precision'], marker='o', label="Precision", linestyle='-')
    plt.plot(labels, df['recall'], marker='o', label="Recall", linestyle='--')
    plt.plot(labels, df['f1-score'], marker='o', label="F1-Score", linestyle=':')
    
    plt.title("Precision, Recall và F1-Score theo các lớp")
    plt.xlabel("Lớp (Label)")
    plt.ylabel("Giá trị")
    plt.xticks(rotation=75, fontsize=8, ha='right')  # Xoay và thu nhỏ font chữ
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Vẽ biểu đồ đường sau khi làm sạch dữ liệu
def classification_report_to_csv_and_plot(report, output_path="classification_report.csv"):
    # Chuyển classification report thành dataframe
    report_data = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Loại bỏ các giá trị không liên quan
            metrics['label'] = label
            report_data.append(metrics)
    
    df = pd.DataFrame(report_data)
    
    # Xóa cột không cần thiết
    if "" in df.columns:
        df = df.drop(columns=[""])  # Xóa cột "" nếu tồn tại
    
    # Lưu kết quả ra file CSV
    df.to_csv(output_path, index=False)
    print(f"Classification Report đã được lưu vào {output_path}")
    
    # Vẽ biểu đồ đường
    df = df[df['label'] != 'accuracy']  # Loại bỏ hàng tổng kết nếu có
    labels = df['label']
    plt.figure(figsize=(20, 8))  # Tăng kích thước để dễ hiển thị hơn
    
    plt.plot(labels, df['precision'], marker='o', label="Precision", linestyle='-')
    plt.plot(labels, df['recall'], marker='o', label="Recall", linestyle='--')
    plt.plot(labels, df['f1-score'], marker='o', label="F1-Score", linestyle=':')
    
    plt.title("Precision, Recall và F1-Score theo các lớp")
    plt.xlabel("Lớp (Label)")
    plt.ylabel("Giá trị")
    plt.xticks(rotation=75, fontsize=8, ha='right')  # Xoay và thu nhỏ font chữ
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# 7. Đánh giá mô hình
y_pred = model.predict(X_test)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))

# Classification Report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Lưu Classification Report vào file CSV và vẽ biểu đồ
classification_report_to_csv_and_plot(report, "classification_report.csv")

# 8. Đánh giá trên tập validation (kiểm tra overfitting)
X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model.fit(X_train_new, y_train_new)  # Huấn luyện lại trên tập nhỏ hơn
y_val_pred = model.predict(X_val)
print("Accuracy on Validation Set:", accuracy_score(y_val, y_val_pred))

# 9. Hàm dự đoán bệnh
def validate_symptoms(symptoms):
    valid_symptoms = label_encoder.classes_
    return [symptom if symptom in valid_symptoms else valid_symptoms[0] for symptom in symptoms]

def predict_disease(symptoms):
    if len(symptoms) < len(symptom_columns):
        symptoms.extend([''] * (len(symptom_columns) - len(symptoms)))

    symptoms = validate_symptoms(symptoms)
    input_data = pd.DataFrame([symptoms], columns=symptom_columns)
    for col in symptom_columns:
        input_data.loc[:, col] = label_encoder.transform(input_data[col])
    
    disease_index = model.predict(input_data)[0]
    disease_name = label_encoder.inverse_transform([disease_index])[0]
    precautions = precaution_data[precaution_data['Disease'] == disease_name].iloc[0, 1:].dropna().values.tolist()
    return disease_name, precautions

# 10. Vẽ biểu đồ tần suất triệu chứng (Đã loại bỏ giá trị rỗng)
all_symptoms = pd.concat([symptom_data[col] for col in symptom_columns])
all_symptoms = all_symptoms[all_symptoms != '']  # Loại bỏ các giá trị rỗng

# Tính tần suất xuất hiện
symptom_counts = all_symptoms.value_counts()

# Vẽ lại biểu đồ
plt.figure(figsize=(12, 6))
symptom_counts[:20].plot(kind='bar', color='orange')
plt.title("Tần suất xuất hiện của các triệu chứng (Đã loại bỏ giá trị rỗng)")
plt.xlabel("Triệu chứng")
plt.ylabel("Số lần xuất hiện")
plt.xticks(rotation=45)
plt.show()

# 11. Biểu đồ phân phối số lượng mẫu của các bệnh
plt.figure(figsize=(12, 6))
sns.countplot(y=symptom_data['Disease'], order=symptom_data['Disease'].value_counts().index, palette='viridis')
plt.title("Phân phối số lượng mẫu của các bệnh")
plt.xlabel("Số lượng mẫu")
plt.ylabel("Tên bệnh")
plt.show()

# 12. Biểu đồ độ chính xác
accuracy = accuracy_score(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='green')
plt.ylim(0, 1)
plt.title("Độ chính xác của mô hình Decision Tree")
plt.ylabel("Accuracy")
plt.show()

# 13. Vẽ ma trận nhầm lẫn với chỉnh sửa trục X
fig, ax = plt.subplots(figsize=(20, 20))  # Tăng kích thước hình để hiển thị rõ hơn
disp = ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    display_labels=label_encoder.classes_,
    cmap='viridis',
    ax=ax
)
plt.title("Ma trận nhầm lẫn của mô hình")
plt.xticks(rotation=90, fontsize=10, ha='right')  # Xoay nhãn trục X 90 độ và căn phải
plt.yticks(fontsize=10)  # Điều chỉnh font chữ cho trục Y
plt.tight_layout()  # Đảm bảo đồ thị không bị cắt
plt.show()

# 14. Vẽ cây quyết định (được cải thiện)
plt.figure(figsize=(25, 15))  # Tăng kích thước biểu đồ
plot_tree(
    model,
    feature_names=symptom_columns,  # Tên các triệu chứng
    class_names=label_encoder.classes_,  # Tên các lớp (bệnh)
    filled=True,  # Tô màu cho các nút
    rounded=True,  # Các nút tròn hơn
    fontsize=8  # Giảm kích thước font để hiển thị vừa phải
)
plt.title("Cây Quyết Định Dự Đoán Bệnh", fontsize=14)  # Tăng kích thước tiêu đề
plt.tight_layout()  # Đảm bảo không gian giữa các thành phần không bị cắt
plt.show()

# 15. Phân tích tầm quan trọng của triệu chứng
feature_importance = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(symptom_columns, feature_importance, color='skyblue')
plt.title("Tầm quan trọng của các triệu chứng")
plt.xlabel("Mức độ quan trọng")
plt.ylabel("Triệu chứng")
plt.show()

# 16. Lưu mô hình để sử dụng sau
joblib.dump(model, "disease_decision_tree_model.pkl")
print("Mô hình đã được lưu vào file 'disease_decision_tree_model.pkl'.")
