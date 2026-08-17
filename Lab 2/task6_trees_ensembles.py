from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from data_loader import load_credit_data


def run_task_6():
    _, _, y_train, y_test, X_tr_num, X_te_num, _, _ = load_credit_data()

    print("=" * 60)
    print("ЗАВДАННЯ 6: Дерева рішень, Boosting та Random Forest")
    print("=" * 60)

    models = {
        "Decision Tree (max_depth=3)": DecisionTreeClassifier(
            max_depth=3, random_state=42
        ),
        "Decision Tree (max_depth=6)": DecisionTreeClassifier(
            max_depth=6, random_state=42
        ),
        "Random Forest (n=50, d=4)": RandomForestClassifier(
            n_estimators=50, max_depth=4, random_state=42
        ),
        "Random Forest (n=150, d=6)": RandomForestClassifier(
            n_estimators=150, max_depth=6, random_state=42
        ),
        "Gradient Boosting (lr=0.05)": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "Gradient Boosting (lr=0.1)": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
    }

    print(f"{'Модель':<32} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 58)
    for name, m in models.items():
        m.fit(X_tr_num, y_train)
        preds = m.predict(X_te_num)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        print(f"{name:<32s} | {acc:<10.4f} | {f1:<10.4f}")


if __name__ == "__main__":
    run_task_6()