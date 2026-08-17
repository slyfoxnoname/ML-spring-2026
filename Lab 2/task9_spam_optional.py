from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


def run_task_9():
    print("=" * 60)
    print("ДОДАТКОВЕ ЗАВДАННЯ 9: Спам-фільтр (Naive Bayes Spam Filtering)")
    print("=" * 60)

    # Демонстраційний корпус SMS (або завантажений spam.csv з Kaggle)
    sample_data = {
        "label": [
            "ham",
            "spam",
            "ham",
            "spam",
            "ham",
            "spam",
            "ham",
            "ham",
            "spam",
            "ham",
        ],
        "text": [
            "Hey, are we still meeting for coffee today?",
            "WINNER! You have won a 1000 cash prize. Call now to claim!",
            "Can you please review the attached document?",
            "URGENT: Your mobile number has won a guaranteed gift.",
            "I will be late by 10 minutes, see you soon.",
            "Free entry into our weekly prize draw! Text WIN to 80085",
            "Let's schedule our sync tomorrow morning.",
            "Thanks for your feedback, appreciated.",
            "Congratulations, you are selected for a free voucher!",
            "Please bring the notebook with you.",
        ],
    }
    df_spam = pd.DataFrame(sample_data)

    X_text = df_spam["text"]
    y_target = df_spam["label"].map({"ham": 0, "spam": 1})

    # Перетворення тексту у вектор частот слів (Bag of Words)
    vec = CountVectorizer(stop_words="english")
    X_features = vec.fit_transform(X_text)

    # Навчання багатовимірного наївного Байєса
    clf = MultinomialNB()
    clf.fit(X_features, y_target)
    preds = clf.predict(X_features)

    print("Звіт класифікації спам-фільтра:")
    print(classification_report(y_target, preds, target_names=["Ham", "Spam"]))


if __name__ == "__main__":
    run_task_9()