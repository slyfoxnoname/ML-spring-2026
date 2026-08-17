import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

from data_loader import load_dataset


def run_task_1():
    X_train, X_test, y_train, y_test, _, _, features = load_dataset()

    print("=" * 65)
    print("ЗАВДАННЯ 1: Побудова ОНК, перевірка кореляцій та якості")
    print("=" * 65)

    # 1. Матриця кореляцій між регресорами
    corr_matrix = X_train.corr()
    print("\n--- Матриця кореляцій між регресорами ---")
    print(corr_matrix.round(4))

    # 2. Оцінка ОНК через statsmodels
    X_train_const = sm.add_constant(X_train)
    ols_stat = sm.OLS(y_train, X_train_const).fit()
    print("\n--- Результати OLS (Statsmodels) ---")
    print(ols_stat.summary())

    # 3. Метрики якості та тест Фішера
    X_test_const = sm.add_constant(X_test)
    y_pred_test = ols_stat.predict(X_test_const)

    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    residual_variance = ols_stat.scale  # Дисперсія помилки (s^2)
    f_stat = ols_stat.fvalue  # F-статистика Фішера
    f_pval = ols_stat.f_pvalue  # p-value тесту Фішера

    print("\n--- Ключові показники якості ОНК ---")
    print(f"Коефіцієнт детермінації (R^2 test): {r2:.4f}")
    print(f"Похибка прогнозування (RMSE test)  : {rmse:.6f}")
    print(f"Дисперсія залишків (s^2)           : {residual_variance:.6f}")
    print(f"Тест Фішера: F-статистика = {f_stat:.2f}, p-value = {f_pval:.4e}")

    # 4. Побудова та збереження діаграми залишків
    residuals_train = ols_stat.resid
    y_pred_train = ols_stat.fittedvalues

    plt.figure(figsize=(8, 5))
    plt.scatter(
        y_pred_train,
        residuals_train,
        alpha=0.6,
        color="royalblue",
        edgecolors="k",
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.title("Діаграма залишків ОНК (Residuals vs Fitted)", fontsize=12)
    plt.xlabel("Розраховані значення (Fitted values)")
    plt.ylabel("Залишки (Residuals)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("residuals_plot.png", dpi=300)
    plt.close()
    print("[+] Графік залишків збережено у 'residuals_plot.png'")

    # 5. Запис детального звіту у файл
    with open("task_1_report.txt", "w", encoding="utf-8") as f:
        f.write("ЗАВДАННЯ 1: ПОВНИЙ ЗВІТ ОНК\n\n")
        f.write("1. МАТРИЦЯ КОРЕЛЯЦІЙ:\n")
        f.write(corr_matrix.round(4).to_string() + "\n\n")
        f.write("2. СТАТИСТИКА OLS:\n")
        f.write(str(ols_stat.summary()) + "\n\n")
        f.write("3. ПІДСУМКОВІ МЕТРИКИ:\n")
        f.write(f"R^2 (тест)        : {r2:.4f}\n")
        f.write(f"RMSE (тест)       : {rmse:.6f}\n")
        f.write(f"Дисперсія залишків: {residual_variance:.6f}\n")
        f.write(f"Тест Фішера (F)   : {f_stat:.2f} (p-val: {f_pval:.4e})\n")


if __name__ == "__main__":
    run_task_1()