
from sklearn.model_selection import train_test_split


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_val, y_train, y_val
