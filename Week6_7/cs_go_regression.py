import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from ydata_profiling import ProfileReport
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("csgo.csv")
# profile = ProfileReport(data, title="CsGo Report", explorative=True)
# profile.to_file("report.html")

# Drop 
cols_to_drop = ['day', 'month', 'year', 'date', 'team_a_rounds', 'team_b_rounds']

# Xoá các cột
data = data.drop(columns=cols_to_drop, axis=1)

# Split target
target = "points"
x = data.drop(target, axis=1)
y = data[target]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)

# Identify feature types
categorical_features = ["map", "result"]
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
    ("regression", LinearRegression())
])

# Train model
model.fit(x_train, y_train)

# Dự đoán
y_predict = model.predict(x_test)

# Đánh giá mô hình
print("MAE: {:.2f}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {:.2f}".format(mean_squared_error(y_test, y_predict)))
print("R2: {:.2f}".format(r2_score(y_test, y_predict)))

# Tạo dữ liệu mẫu để dự đoán
new_data = pd.DataFrame([[
    'Mirage',     # map
    3,            # day
    8,            # month
    2018,         # year
    '3/8/2018',   # date (nên xử lý datetime nếu cần)
    327,          # wait_time_s
    2906,         # match_time_s
    16,           # team_a_rounds
    13,           # team_b_rounds
    215,          # ping
    17,           # kills
    2,            # assists
    21,           # deaths
    2,            # mvps
    5,            # hs_percent
    'Win'         # result
]], columns=[
    'map', 'day', 'month', 'year', 'date',
    'wait_time_s', 'match_time_s', 'team_a_rounds', 'team_b_rounds',
    'ping', 'kills', 'assists', 'deaths', 'mvps',
    'hs_percent', 'result'
])

predicted_points = model.predict(new_data)

print("Dự đoán point: {:.2f}".format(predicted_points[0]))