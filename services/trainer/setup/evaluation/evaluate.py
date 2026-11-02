from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)

    return {
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, pos_label=">50K"),
        "recall": recall_score(y_val, predictions, pos_label=">50K"),
        "f1": f1_score(y_val, predictions, pos_label=">50K"),
    }
