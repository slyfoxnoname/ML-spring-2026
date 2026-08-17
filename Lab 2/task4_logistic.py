from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from data_loader import load_credit_data


def run_task_4():
    _, _, y_train, y_test, X_tr_num, X_te_num, _, _ = load_credit_data()

    print("=" * 60)
    print("ЗАВДАННЯ 4: Логістична регресія")
    print("=" * 60)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_tr_num, y_train)
    y_pred = clf.predict(X_te_num)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}\n")
    print("Детальний звіт класифікації:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    run_task_4()