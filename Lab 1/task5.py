import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def generate_data(N: int, beta: list) -> pd.DataFrame:
    X_syn = np.random.normal(0, 1, size=(N, 3))
    eps = np.random.normal(0, 0.1, size=N)
    y_syn = (
        beta[0]
        + beta[1] * X_syn[:, 0]
        + beta[2] * X_syn[:, 1]
        + beta[3] * X_syn[:, 2]
        + eps
    )
    df_syn = pd.DataFrame(X_syn, columns=["X1", "X2", "X3"])
    df_syn["Y"] = y_syn
    return df_syn


def ols_normal_eq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y


def ols_gd(
    X: np.ndarray, y: np.ndarray, lr: float = 0.01, steps: int = 250
) -> np.ndarray:
    N = X.shape[0]
    X_b = np.c_[np.ones((N, 1)), X]
    w = np.zeros(X_b.shape[1])
    for _ in range(steps):
        grad = (2 / N) * X_b.T @ (X_b @ w - y)
        w -= lr * grad
    return w


def run_task_5():
    print("=" * 60)
    print("ЗАВДАННЯ 5: Порівняння часу обчислення ОНК трьома методами")
    print("=" * 60)

    beta_true = [1.5, 0.8, -1.2, 0.3]
    sample_sizes = [1000, 10000, 100000]
    benchmark_rows = []

    for N in sample_sizes:
        df_gen = generate_data(N, beta_true)
        X = df_gen[["X1", "X2", "X3"]].to_numpy()
        y = df_gen["Y"].to_numpy()

        times_norm, times_sk, times_gd = [], [], []

        for _ in range(10):  # 10 повторів
            # 1. Normal Equation
            t0 = time.perf_counter()
            ols_normal_eq(X, y)
            times_norm.append(time.perf_counter() - t0)

            # 2. Sklearn
            t0 = time.perf_counter()
            LinearRegression().fit(X, y)
            times_sk.append(time.perf_counter() - t0)

            # 3. Gradient Descent
            t0 = time.perf_counter()
            ols_gd(X, y, steps=250)
            times_gd.append(time.perf_counter() - t0)

        mean_norm = np.mean(times_norm)
        mean_sk = np.mean(times_sk)
        mean_gd = np.mean(times_gd)

        benchmark_rows.append(
            {
                "N": N,
                "Normal_Eq_sec": mean_norm,
                "Sklearn_sec": mean_sk,
                "GD_sec": mean_gd,
            }
        )

        print(
            f"N = {N:6d} | Normal Eq: {mean_norm:.5f}s | Sklearn: {mean_sk:.5f}s | GD: {mean_gd:.5f}s"
        )

    # Збереження результатів
    df_res = pd.DataFrame(benchmark_rows)
    df_res.to_csv("task_5_benchmark.csv", index=False)
    print("\n[+] Таблицю часу збережено у task_5_benchmark.csv")


if __name__ == "__main__":
    run_task_5()