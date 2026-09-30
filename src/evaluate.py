import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

class_names = ["Normal", "Suspect", "Pathological"]
results = []  # shared list that collects every model's metrics


def evaluate(name, y_true, y_pred):
    """Store macro metrics, print the report and save a confusion matrix."""
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
    })
    print(classification_report(y_true, y_pred, target_names=class_names))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                           display_labels=class_names).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    plt.savefig(f"../results/figures/cm_{name.replace(' ', '_')}.png", dpi=150)
    plt.show()