import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Đọc dữ liệu
symptom_data = pd.read_csv("DiseaseAndSymptoms.csv")
symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]
label_encoder = LabelEncoder()

# 2. Xử lý dữ liệu
symptom_data = symptom_data.fillna('')

# Tạo danh sách tất cả các triệu chứng duy nhất
all_symptoms = pd.concat([symptom_data[col] for col in symptom_columns]).unique()
all_symptoms = [symptom for symptom in all_symptoms if symptom != '']

# Mã hóa nhãn bệnh
symptom_data['Disease'] = label_encoder.fit_transform(symptom_data['Disease'])

# 3. Hàm tính Entropy
def calculate_entropy(data, target_column='Disease'):
    total_size = len(data)
    if total_size == 0:
        return 0
    probabilities = data[target_column].value_counts() / total_size
    entropy = -sum(probabilities * np.log2(probabilities))
    return entropy

# 4. Hàm tính Gini
def calculate_gini(data, target_column='Disease'):
    total_size = len(data)
    if total_size == 0:
        return 0
    probabilities = data[target_column].value_counts() / total_size
    gini = 1 - sum(probabilities**2)
    return gini

# 5. Hàm tính IR cho từng triệu chứng
def calculate_metrics(symptom, data, target_column='Disease'):
    # Chia dữ liệu theo triệu chứng
    has_symptom = data[data.isin([symptom]).any(axis=1)]
    no_symptom = data[~data.isin([symptom]).any(axis=1)]
    
    # Tính Entropy và Gini trước
    entropy_before = calculate_entropy(data, target_column)
    gini_before = calculate_gini(data, target_column)
    
    # Tính Entropy và Gini sau khi chia
    total_size = len(data)
    entropy_after, gini_after = 0, 0
    for subset in [has_symptom, no_symptom]:
        size = len(subset)
        if size > 0:
            weight = size / total_size
            entropy_after += weight * calculate_entropy(subset, target_column)
            gini_after += weight * calculate_gini(subset, target_column)
    
    # Tính IR
    ir_entropy = entropy_before - entropy_after
    ir_gini = gini_before - gini_after
    
    return {
        'Symptom': symptom,
        'Entropy Before': entropy_before,
        'Entropy After': entropy_after,
        'Gini Before': gini_before,
        'Gini After': gini_after,
        'IR Entropy': ir_entropy,
        'IR Gini': ir_gini
    }

# 6. Tính các giá trị cho tất cả triệu chứng
results = []
for symptom in all_symptoms:
    metrics = calculate_metrics(symptom, symptom_data)
    results.append(metrics)

# 7. Lưu kết quả ra file CSV
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='IR Entropy', ascending=False)
results_df.to_csv("symptom_metrics_results.csv", index=False)

# Vẽ biểu đồ
top_n = 20  # Số triệu chứng hàng đầu
top_results = results_df.head(top_n)

# Vẽ biểu đồ IR Entropy
plt.figure(figsize=(12, 6))
plt.bar(top_results['Symptom'], top_results['IR Entropy'], color='blue', alpha=0.7, label='IR Entropy')
plt.title("IR Entropy của Top 20 Triệu Chứng")
plt.xlabel("Triệu Chứng")
plt.ylabel("IR Entropy")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Vẽ biểu đồ IR Gini
plt.figure(figsize=(12, 6))
plt.bar(top_results['Symptom'], top_results['IR Gini'], color='green', alpha=0.7, label='IR Gini')
plt.title("IR Gini của Top 20 Triệu Chứng")
plt.xlabel("Triệu Chứng")
plt.ylabel("IR Gini")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

print("Kết quả đã được lưu vào file 'symptom_metrics_results.csv'")
