import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Student Performance Prediction")

data = pd.read_csv("student-mat.csv", sep=";")
print(data)

features = ["studytime", "absences", "G1", "G2"]
X = data[features]
y = data["G3"]

print("\nDataset loaded:", data.shape)
print("Missing values:", data.isnull().sum().sum())
print("Target: G3 (Final Grade)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 2))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Student Performance")
plt.plot([0, 20], [0, 20], linestyle="--")
plt.grid()
plt.show()

print("\nEnter New Student Details")

name = input("Name: ")
studytime = float(input("Study time: "))
absences = float(input("Absences: "))
g1 = float(input("G1 (First period grade 0-20): "))
g2 = float(input("G2 (Second period grade 0-20): "))

student = pd.DataFrame({
    "studytime": [studytime],
    "absences": [absences],
    "G1": [g1],
    "G2": [g2]
})

g3 = np.clip(model.predict(student)[0], 0, 20)

print("\nPrediction Result")
print("Name:", name)
print("G1:", g1)
print("G2:", g2)
print("Predicted G3:", round(g3, 2))