from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB
from data_loader import load_credit_data


def run_task_2():
    _, _, y_train, y_test, X_tr_num, X_te_num, _, _ = load_credit_data()

    print("=" * 60)
    print("ЗАВДАННЯ 2: Наївний байєсівський класифікатор (GaussianNB)")
    print("=" * 60)

    model = GaussianNB()
    model.fit(X_tr_num, y_train)
    y_pred = model.predict(X_te_num)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Точність (Accuracy) : {acc:.4f}")
    print(f"F1-Score            : {f1:.4f}")
    print("\nМатриця помилок (Confusion Matrix):")
    print(cm)


if __name__ == "__main__":
    run_task_2()