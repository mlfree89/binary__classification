
from .loader import COLUMNS


def validate_data(df):
    if list(df.columns) != COLUMNS:
        raise ValueError("Unexpected dataset columns")

    if df.empty:
        raise ValueError("Dataset is empty")

    if df["income"].isna().any():
        raise ValueError("Target column contains missing values")

    valid_labels = {"<=50K", ">50K"}
    labels = set(df["income"].dropna().unique())

    if not labels.issubset(valid_labels):
        raise ValueError(f"Unexpected income labels: {labels}")

    return True


if __name__ == "__main__":
    from .loader import load_data

    train, test = load_data()
    print(validate_data(train))
    print(validate_data(test))
