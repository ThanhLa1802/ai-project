import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

def create_ts_data(data, window_size):
    i = 1
    while i < window_size:
        data["Demand_{}".format(i)] = data["Demand"].shift(-i)
        i += 1
    data["target"] = data["Demand"].shift(-i)
    data = data.dropna(axis=0)
    return data

data = pd.read_csv("electricity.csv")
data["Time"] = pd.to_datetime(data["Time"])
data["Demand"] = data["Demand"].interpolate()

# Plot Figure
# fig, ax = plt.subplots()
# ax.plot(data["Time"], data["Demand"])
# ax.set_xlabel("Time")
# ax.set_ylabel("Demand")
# plt.show()

window_size = 5
train_ratio = 0.8
data = create_ts_data(data, window_size)

x = data.drop(["Time", "target", "Temperature", "Date"], axis=1)
print(x.columns)
y = data["target"]
num_samples = len(data)
x_train = x[:int(num_samples*train_ratio)]
y_train = y[:int(num_samples*train_ratio)]
x_test = x[int(num_samples*train_ratio):]
y_test = y[int(num_samples*train_ratio):]

model = LinearRegression()

model.fit(x_train, y_train)
#filename = 'model.pkl'
# Save model
#pickle.dump(model, open(filename, 'wb'))
y_predict = model.predict(x_test)
for i, j in zip(y_predict, y_test):
    print("Predicted value: {}. Actual value: {}".format(i, j))
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))
fig, ax = plt.subplots()
ax.plot(data["Time"][:int(num_samples*train_ratio)], y_train, label="train")
ax.plot(data["Time"][int(num_samples*train_ratio):], y_test, label="test")
ax.plot(data["Time"][int(num_samples*train_ratio):], y_predict, label="prediction")
ax.set_xlabel("Time")
ax.set_ylabel("Demand")
ax.legend()
ax.grid()
plt.show()
# Load model
# model = pickle.load(open(filename, 'rb'))
current_data = [3304.641186, 3310.641186, 3909.300738, 3900.300738, 3809.300738, 3809.300738]
prediction = model.predict([current_data])
print(prediction)

