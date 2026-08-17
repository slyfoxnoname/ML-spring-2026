from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from data_loader import load_credit_data


def run_task_3():
    _, _, y_train, y_test, X_tr_num, X_te_num, _, _ = load_credit_data()

    print("=" * 60)
    print("ЗАВДАННЯ 3: Класифікація методом k-NN")
    print("=" * 60)

    k_values = [3, 5, 7, 9, 11, 15, 21]
    best_k, best_acc, best_f1 = None, 0, 0

    print(f"{'k (сусідів)':<12} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 38)
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr_num, y_train)
        preds = knn.predict(X_te_num)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        print(f"{k:<12d} | {acc:<10.4f} | {f1:<10.4f}")

        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_k = k

    print(f"\nНайкращий результат: k = {best_k} (Accuracy = {best_acc:.4f})")


if __name__ == "__main__":
    run_task_3()