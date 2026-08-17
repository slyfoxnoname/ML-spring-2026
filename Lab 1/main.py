import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import statsmodels.api as sm

# ----------------------------------------------------
# 0. Читання та підготовка вибірки
# ----------------------------------------------------
df = pd.read_csv("NAICExpense.csv")

features = ["RBC", "STAFFWAGE", "AGENTWAGE", "LONGLOSS", "SHORTLOSS"]
target = "EXPENSES"

# Видаляємо пропуски в даних
df_clean = df[[target] + features].dropna()

X = df_clean[features]
y = df_clean[target]

# Розбивка 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Масштабування
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Словник для фінальної зведеної таблиці
summary_metrics = []


# ----------------------------------------------------
# 1. Звичайний МНК (OLS)
# ----------------------------------------------------
X_train_const = sm.add_constant(X_train)
ols_stat = sm.OLS(y_train, X_train_const).fit()
ols_summary_str = str(ols_stat.summary())

ols_sk = LinearRegression()
ols_sk.fit(X_train, y_train)
y_pred_ols = ols_sk.predict(X_test)

r2_ols = r2_score(y_test, y_pred_ols)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
mae_ols = mean_absolute_error(y_test, y_pred_ols)

summary_metrics.append(
    {"Модель": "1. Базовий МНК (OLS)", "R2": r2_ols, "RMSE": rmse_ols, "MAE": mae_ols}
)


# ----------------------------------------------------
# 2. Гребенева регресія (Ridge)
# ----------------------------------------------------
ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

summary_metrics.append(
    {
        "Модель": f"2. Ridge (alpha={ridge.alpha_:.3f})",
        "R2": r2_ridge,
        "RMSE": rmse_ridge,
        "MAE": mae_ridge,
    }
)


# ----------------------------------------------------
# 3. Нелінійність (поліноми 2-го степеня)
# ----------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_p = poly.fit_transform(X_train_scaled)
X_test_p = poly.transform(X_test_scaled)

poly_lr = LinearRegression()
poly_lr.fit(X_train_p, y_train)
y_pred_poly = poly_lr.predict(X_test_p)

r2_poly = r2_score(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
mae_poly = mean_absolute_error(y_test, y_pred_poly)

summary_metrics.append(
    {
        "Модель": "3. Поліноміальна (deg=2)",
        "R2": r2_poly,
        "RMSE": rmse_poly,
        "MAE": mae_poly,
    }
)


# ----------------------------------------------------
# 4. Скорочення ознак (Lasso та фільтрація)
# ----------------------------------------------------
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)
lasso_coefs = dict(zip(features, lasso.coef_))

# Модель лише на значущих регресорах
best_features = ["RBC", "LONGLOSS", "SHORTLOSS"]
ols_red = LinearRegression().fit(X_train[best_features], y_train)
y_pred_red = ols_red.predict(X_test[best_features])

r2_red = r2_score(y_test, y_pred_red)
rmse_red = np.sqrt(mean_squared_error(y_test, y_pred_red))
mae_red = mean_absolute_error(y_test, y_pred_red)

summary_metrics.append(
    {
        "Модель": "4. Скорочена (3 регресори)",
        "R2": r2_red,
        "RMSE": rmse_red,
        "MAE": mae_red,
    }
)


# ----------------------------------------------------
# 5. Генерація та бенчмарк часу
# ----------------------------------------------------
def make_data(n, b):
    x_mat = np.random.normal(0, 1, size=(n, 3))
    err = np.random.normal(0, 0.1, size=n)
    y_vec = b[0] + b[1] * x_mat[:, 0] + b[2] * x_mat[:, 1] + b[3] * x_mat[:, 2] + err
    return x_mat, y_vec


def calc_normal_eq(x, y_val):
    x_b = np.c_[np.ones((x.shape[0], 1)), x]
    return np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y_val


def calc_gd(x, y_val, lr=0.01, steps=250):
    n_rows = x.shape[0]
    x_b = np.c_[np.ones((n_rows, 1)), x]
    w = np.zeros(x_b.shape[1])
    for _ in range(steps):
        grad = (2 / n_rows) * x_b.T @ (x_b @ w - y_val)
        w -= lr * grad
    return w


betas = [1.5, 0.8, -1.2, 0.3]
sample_sizes = [1000, 10000, 100000]
benchmark_res = []

for n in sample_sizes:
    x_arr, y_arr = make_data(n, betas)
    t_norm, t_sk, t_gd = [], [], []

    for _ in range(10):
        # 1. Normal Eq
        t0 = time.perf_counter()
        calc_normal_eq(x_arr, y_arr)
        t_norm.append(time.perf_counter() - t0)

        # 2. Sklearn
        t0 = time.perf_counter()
        LinearRegression().fit(x_arr, y_arr)
        t_sk.append(time.perf_counter() - t0)

        # 3. Gradient Descent
        t0 = time.perf_counter()
        calc_gd(x_arr, y_arr, steps=250)
        t_gd.append(time.perf_counter() - t0)

    benchmark_res.append(
        {
            "N": n,
            "Normal_Eq_sec": np.mean(t_norm),
            "Sklearn_sec": np.mean(t_sk),
            "GD_sec": np.mean(t_gd),
        }
    )


# ----------------------------------------------------
# ФУНКЦІЯ ЗАПИСУ РЕЗУЛЬТАТІВ У ФАЙЛИ
# ----------------------------------------------------
def save_results_to_file(txt_path="regression_report.txt", csv_path="metrics_summary.csv"):
    # 1. Зберігаємо таблицю метрик у CSV для Excel
    df_metrics = pd.DataFrame(summary_metrics)
    df_metrics.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 2. Формуємо красивий текстовий звіт
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("         ЗВІТ З ЛАБОРАТОРНОЇ РОБОТИ (РЕГРЕСІЙНИЙ АНАЛІЗ)\n")
        f.write("=" * 65 + "\n\n")

        f.write("1. ПОРІВНЯЛЬНА ТАБЛИЦЯ ЯКОСТІ МОДЕЛЕЙ:\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Модель':<30} | {'R2':<8} | {'RMSE':<10} | {'MAE':<10}\n")
        f.write("-" * 65 + "\n")
        for m in summary_metrics:
            f.write(
                f"{m['Модель']:<30} | {m['R2']:<8.4f} | {m['RMSE']:<10.5f} | {m['MAE']:<10.5f}\n"
            )
        f.write("-" * 65 + "\n\n")

        f.write("2. КОЕФІЦІЄНТИ LASSO (ВІДБІР ОЗНАК):\n")
        f.write("-" * 65 + "\n")
        for k, v in lasso_coefs.items():
            status = "ЗБЕРЕЖЕНО" if abs(v) > 1e-5 else "ОБНУЛЕНО (зайвий)"
            f.write(f"  {k:<12}: {v:>10.6f}  [{status}]\n")
        f.write("\n")

        f.write("3. БЕНЧМАРК ШВИДКОСТІ ОБЧИСЛЕННЯ ОНК (ПУНКТ 5):\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Розмір (N)':<12} | {'Normal Eq':<12} | {'Sklearn':<12} | {'Grad Descent':<12}\n")
        f.write("-" * 65 + "\n")
        for b in benchmark_res:
            f.write(
                f"{b['N']:<12d} | {b['Normal_Eq_sec']:<10.5f}s  | {b['Sklearn_sec']:<10.5f}s  | {b['GD_sec']:<10.5f}s\n"
            )
        f.write("-" * 65 + "\n\n")

        f.write("4. ДЕТАЛЬНА СТАТИСТИКА OLS (STATMODELS):\n")
        f.write("-" * 65 + "\n")
        f.write(ols_summary_str + "\n")

    print(f"[+] Результати успішно записано у файли:")
    print(f"    - {txt_path} (текстовий звіт)")
    print(f"    - {csv_path} (зведена таблиця)")


# Викликаємо функцію збереження
if __name__ == "__main__":
    save_results_to_file()