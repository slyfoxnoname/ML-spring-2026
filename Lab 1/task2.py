import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import load_dataset


def run_task_2():
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        _,
    ) = load_dataset()

    print("=" * 65)
    print("ЗАВДАННЯ 2: Гребенева регресія (Ridge, підбір параметра lambda)")
    print("=" * 65)

    alphas = np.logspace(-3, 3, 50)
    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_train_scaled, y_train)

    y_pred_train = ridge.predict(X_train_scaled)
    y_pred_test = ridge.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    residual_variance_ridge = np.var(y_train - y_pred_train, ddof=1)

    print(f"Оптимальний параметр lambda (alpha): {ridge.alpha_:.4f}")
    print(f"R^2 Score (тест)                   : {r2:.4f}")
    print(f"RMSE (тест)                        : {rmse:.6f}")
    print(f"Дисперсія залишків моделі Ridge    : {residual_variance_ridge:.6f}")

    with open("task_2_report.txt", "w", encoding="utf-8") as f:
        f.write("ЗАВДАННЯ 2: РЕЗУЛЬТАТИ RIDGE\n\n")
        f.write(f"Оптимальний lambda (alpha): {ridge.alpha_:.4f}\n")
        f.write(f"R^2 (тест)                : {r2:.4f}\n")
        f.write(f"RMSE (тест)               : {rmse:.6f}\n")
        f.write(f"Дисперсія залишків        : {residual_variance_ridge:.6f}\n")


if __name__ == "__main__":
    run_task_2()