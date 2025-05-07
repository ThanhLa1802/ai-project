import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Đọc dữ liệu
data = pd.read_csv("car.csv")
target = "mpg"

# Loại bỏ cột không cần thiết
data = data.drop("car name", axis=1)

# Xử lý cột 'horsepower' sang dạng số, thay '?' bằng NaN
data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")

# Chia dữ liệu thành features và label
x = data.drop(target, axis=1)
y = data[target]

# Chia dữ liệu train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)

# Các cột số
numeric_features = ["displacement", "horsepower", "weight", "acceleration"]
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False))
])
# Tiền xử lý
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, numeric_features),
    ("nom_features", nom_transformer, ["cylinders", "origin", "model year"])
])

# Pipeline gồm tiền xử lý và mô hình
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Huấn luyện mô hình
model.fit(x_train, y_train)

# Dự đoán
y_predict = model.predict(x_test)

# Đánh giá mô hình
print("MAE: {:.2f}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {:.2f}".format(mean_squared_error(y_test, y_predict)))
print("R2: {:.2f}".format(r2_score(y_test, y_predict)))

new_data = pd.DataFrame([[
    8,          # cylinders
    120.0,      # displacement
    88.0,       # horsepower
    2500,       # weight
    15.0,       # acceleration
    82,         # model year
    2           # origin
]], columns=["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"])

# Dự đoán MPG cho dữ liệu mới
predicted_mpg = model.predict(new_data)

print("Dự đoán MPG cho xe mới là: {:.2f}".format(predicted_mpg[0]))