import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load Dataset
data = pd.read_csv('dataset.csv')  # Pastikan path file benar

# 2. Membuat kolom target 'UCL_Eligible' berdasarkan kolom 'LgRk'
data['UCL_Eligible'] = data['LgRk'].apply(lambda x: 1 if x <= 4 else 0)

# 3. Pilih Fitur yang Digunakan
features = ['Pts/G', 'xG', 'xGA', 'xGD', 'xGD/90', 'W']  # Fitur yang digunakan untuk prediksi

# 4. Konversi Kolom Kategorikal menjadi Numerik jika ada
labelencoder = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = labelencoder.fit_transform(data[col])

# 5. Pisahkan Fitur (X) dan Label (y)
X = data[features]  # Fitur yang dipilih
y = data['UCL_Eligible']  # Kolom 'UCL_Eligible' sebagai label

# 6. Split Dataset (Training dan Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'models/rf_model.pkl')  # Simpan model

# 8. Train C4.5 (Decision Tree)
print("Training C4.5 (Decision Tree)...")
c45_model = DecisionTreeClassifier(criterion='entropy')
c45_model.fit(X_train, y_train)
joblib.dump(c45_model, 'models/c45_model.pkl')  # Simpan model

# 9. Train XGBoost
print("Training XGBoost...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'models/xgb_model.pkl')  # Simpan model

# 10. Evaluasi Akurasi
print("\nEvaluasi Model:")
for name, model in [("Random Forest", rf_model), ("C4.5", c45_model), ("XGBoost", xgb_model)]:
    predictions = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, predictions):.2f}")
