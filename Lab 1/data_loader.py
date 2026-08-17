from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["RBC", "STAFFWAGE", "AGENTWAGE", "LONGLOSS", "SHORTLOSS"]
TARGET = "EXPENSES"

# Визначаємо шлях до поточної папки, де лежить data_loader.py
BASE_DIR = Path(__file__).resolve().parent


def load_dataset(csv_filename="NAICExpense.csv"):
    csv_path = BASE_DIR / csv_filename

    # Якщо раптом файл лежить на рівень вище у папці zxc
    if not csv_path.exists():
        csv_path = BASE_DIR.parent / csv_filename

    df = pd.read_csv(csv_path)
    clean_df = df[[TARGET] + FEATURES].dropna()

    X = clean_df[FEATURES]
    y = clean_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        FEATURES,
    )