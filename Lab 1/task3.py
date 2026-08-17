import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from data_loader import load_dataset


def run_task_3():
    _, _, y_train, y_test, X_train_scaled, X_test_scaled, _ = load_dataset()

    print("=" * 60)
    print("ЗАВДАННЯ 3: Додавання нелінійності (поліноми 2-го степеня)")
    print("=" * 60)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    poly_lr = LinearRegression()
    poly_lr.fit(X_train_poly, y_train)
    y_pred = poly_lr.predict(X_test_poly)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Кількість ознак після поліномів: {X_train_poly.shape[1]}")
    print(f"R^2 Score (на тесті)           : {r2:.4f}")
    print(f"RMSE (на тесті)                : {rmse:.6f}")

    with open("task_3_report.txt", "w", encoding="utf-8") as f:
        f.write("ЗАВДАННЯ 3: ПОЛІНОМІАЛЬНА РЕГРЕСІЯ\n\n")
        f.write(f"Кількість ознак: {X_train_poly.shape[1]}\n")
        f.write(f"R^2 Score      : {r2:.4f}\n")
        f.write(f"RMSE           : {rmse:.6f}\n")
    print("\n[+] Результати збережено у task_3_report.txt")


if __name__ == "__main__":
    run_task_3()