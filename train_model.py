import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
df = pd.read_csv('garments_worker_productivity.csv')

# 2. Pre-processing
df['wip'] = df['wip'].fillna(0)
df['department'] = df['department'].str.strip()
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df = df.drop(['date'], axis=1)

# Categorical Encoding
le = LabelEncoder()
for col in ['quarter', 'department', 'day']:
    df[col] = le.fit_transform(df[col])

# 3. Split Data
X = df.drop('actual_productivity', axis=1)
y = df['actual_productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save Model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully as model.pkl")