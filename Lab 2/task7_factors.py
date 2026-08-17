import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from data_loader import load_credit_data


def run_task_7():
    X_train_raw, X_test_raw, y_train, y_test, _, _, num_cols, cat_cols = (
        load_credit_data()
    )

    print("=" * 60)
    print("ЗАВДАННЯ 7: Дослідження впливу факторних даних (A13 та інші)")
    print("=" * 60)

    # 1. Підхід A: Кодування One-Hot Encoding для всіх факторів
    df_full_X = pd.concat([X_train_raw, X_test_raw])
    df_encoded = pd.get_dummies(df_full_X, columns=cat_cols, drop_first=True)

    X_tr_enc = df_encoded.iloc[: len(X_train_raw)]
    X_te_enc = df_encoded.iloc[len(X_train_raw) :]

    rf_full = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42
    )
    rf_full.fit(X_tr_enc, y_train)
    preds_full = rf_full.predict(X_te_enc)

    print("--- Підхід А: One-Hot кодування всіх факторів ---")
    print(f"Кількість ознак після OHE : {X_tr_enc.shape[1]}")
    print(f"Accuracy на тесті         : {accuracy_score(y_test, preds_full):.4f}")
    print(f"F1-Score на тесті         : {f1_score(y_test, preds_full):.4f}\n")

    # 2. Підхід Б: Аналіз підвибірок за фактором A13 (значення 'g', 'p', 's')
    print("--- Підхід Б: Поділ вибірки за значенням фактора A13 ---")
    for val in X_train_raw["A13"].unique():
        idx_tr = X_train_raw["A13"] == val
        idx_te = X_test_raw["A13"] == val

        if idx_tr.sum() > 20 and idx_te.sum() > 5:
            rf_sub = RandomForestClassifier(
                n_estimators=50, max_depth=4, random_state=42
            )
            rf_sub.fit(X_tr_enc[idx_tr], y_train[idx_tr])
            preds_sub = rf_sub.predict(X_te_enc[idx_te])
            print(
                f"Підвибірка A13='{val}' (Train N={idx_tr.sum()}, Test N={idx_te.sum()}): Accuracy = {accuracy_score(y_test[idx_te], preds_sub):.4f}"
            )


if __name__ == "__main__":
    run_task_7()