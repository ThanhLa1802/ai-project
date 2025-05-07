import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from ydata_profiling import ProfileReport


data = pd.read_csv("csgo.csv")
# profile = ProfileReport(data, title="CsGo Report", explorative=True)
# profile.to_file("report.html")

# Drop 
cols_to_drop = ['day', 'month', 'year', 'date', 'team_a_rounds', 'team_b_rounds']

# Xoá các cột
data = data.drop(columns=cols_to_drop, axis=1)

# Split target
target = "result"
x = data.drop(target, axis=1)
y = data[target]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)

# Identify feature types
categorical_features = ["map"]
numeric_features = [col for col in x.columns if col not in categorical_features]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Full pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

# Train model
model.fit(x_train, y_train)

# Predict
y_predict = model.predict(x_test)

# Evaluate
print(classification_report(y_test, y_predict))