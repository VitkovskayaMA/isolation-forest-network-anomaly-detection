import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import RobustScaler

print("Загрузка данных")
train = pd.read_csv('DataSet/UNSW_NB15_training-set.csv')
test = pd.read_csv('DataSet/UNSW_NB15_testing-set.csv')

train = train.drop(columns=['id', 'attack_cat'], errors='ignore')
test = test.drop(columns=['id', 'attack_cat'], errors='ignore')

categorical_columns = ['proto', 'service', 'state']
combined = pd.concat([train, test], axis=0)
combined_encoded = pd.get_dummies(combined, columns=categorical_columns, drop_first=True)

train_df = combined_encoded.iloc[:len(train), :].copy()
test_df = combined_encoded.iloc[len(train):, :].copy()

X_train_normal = train_df[train_df['label'] == 0].drop('label', axis=1)
X_test = test_df.drop('label', axis=1)
y_test = test_df['label'].astype(int)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)

print("Обучение финальной модели")
model = IsolationForest(
    n_estimators=200, 
    contamination=0.3, 
    random_state=42, 
    n_jobs=-1
)
model.fit(X_train_scaled)


y_pred = np.where(model.predict(X_test_scaled) == -1, 1, 0)

print(" РЕЗУЛЬТАТЫ")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nОтчет классификации:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix: Isolation Forest Anomaly Detection')
plt.xlabel('Предсказано моделью')
plt.ylabel('Данные')
plt.show()