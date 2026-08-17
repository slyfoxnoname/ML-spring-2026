from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from data_loader import load_credit_data


def run_task_5():
    _, _, y_train, y_test, X_tr_num, X_te_num, _, _ = load_credit_data()

    print("=" * 60)
    print("ЗАВДАННЯ 5: SVM з підбором ядер та гіперпараметрів")
    print("=" * 60)

    configs = [
        {"kernel": "linear", "C": 0.1},
        {"kernel": "linear", "C": 1.0},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 5.0, "gamma": 0.1},
        {"kernel": "poly", "C": 1.0, "degree": 2},
        {"kernel": "poly", "C": 1.0, "degree": 3},
    ]

    print(
        f"{'Конфігурація SVM':<32} | {'Accuracy':<10} | {'F1-Score':<10}"
    )
    print("-" * 58)

    for cfg in configs:
        svm = SVC(**cfg, random_state=42)
        svm.fit(X_tr_num, y_train)
        preds = svm.predict(X_te_num)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cfg_str = str(cfg)
        print(f"{cfg_str:<32s} | {acc:<10.4f} | {f1:<10.4f}")


if __name__ == "__main__":
    run_task_5()