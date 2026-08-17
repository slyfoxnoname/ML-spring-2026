import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import load_dataset


def run_task_4():
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        features,
    ) = load_dataset()

    print("=" * 60)
    print("ЗАВДАННЯ 4: Зменшення кількості регресорів")
    print("=" * 60)

    # 1. Відбір через Lasso
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train_scaled, y_train)

    print("Коефіцієнти Lasso:")
    for name, coef in zip(features, lasso.coef_):
        status = "Залишено" if abs(coef) > 1e-5 else "Обнулено (незначущий)"
        print(f"  {name:<10}: {coef:>10.6f} -> {status}")

    # 2. Перенавчання без STAFFWAGE та AGENTWAGE
    selected_features = ["RBC", "LONGLOSS", "SHORTLOSS"]
    reduced_model = LinearRegression().fit(X_train[selected_features], y_train)
    y_pred_red = reduced_model.predict(X_test[selected_features])

    r2_red = r2_score(y_test, y_pred_red)
    rmse_red = np.sqrt(mean_squared_error(y_test, y_pred_red))

    print(
        f"\nЯкість моделі лише на 3 регресорах ({', '.join(selected_features)}):"
    )
    print(f"R^2 Score : {r2_red:.4f}")
    print(f"RMSE      : {rmse_red:.6f}")

    with open("task_4_report.txt", "w", encoding="utf-8") as f:
        f.write("ЗАВДАННЯ 4: ВІДБІР ОЗНАК (LASSO)\n\n")
        for name, coef in zip(features, lasso.coef_):
            f.write(f"{name:<10}: {coef:.6f}\n")
        f.write(
            f"\nРезультат після скорочення до 3 змінних: R^2 = {r2_red:.4f}, RMSE = {rmse_red:.6f}\n"
        )
    print("\n[+] Результати збережено у task_4_report.txt")


if __name__ == "__main__":
    run_task_4()