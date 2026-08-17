from pathlib import Path
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

NUM_COLS = ["A2", "A3", "A8", "A11", "A14", "A15"]
CAT_COLS = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
ALL_COLS = [f"A{i}" for i in range(1, 17)]
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"


def load_credit_data(filename="crx.data"):
    filepath = BASE_DIR / filename
    if not filepath.exists():
        filepath = BASE_DIR.parent / filename

    # Якщо файлу немає локально — завантажуємо автоматично з репозиторію UCI
    if not filepath.exists():
        target_path = BASE_DIR / filename
        print(f"[*] Завантаження {filename} з репозиторію UCI...")
        urllib.request.urlretrieve(DATA_URL, target_path)
        filepath = target_path

    df = pd.read_csv(filepath, header=None, names=ALL_COLS, na_values="?")

    # Обробка пропусків: числові -> середнє, категоріальні -> мода
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())

    for col in CAT_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Бінаризація цільової змінної (+ -> 1, - -> 0)
    df["A16"] = df["A16"].map({"+": 1, "-": 0})

    X = df.drop(columns=["A16"])
    y = df["A16"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[NUM_COLS])
    X_test_num_scaled = scaler.transform(X_test[NUM_COLS])

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_num_scaled,
        X_test_num_scaled,
        NUM_COLS,
        CAT_COLS,
    )