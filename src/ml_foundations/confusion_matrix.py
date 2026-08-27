from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def get_predictions(count: int = 10) -> tuple:
    dataset = load_breast_cancer()

    X = dataset.data
    y = (dataset.target == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(max_depth=2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test[:count], y_pred[:count]


def main() -> None:
    y_true, y_pred = get_predictions()

    print("# | y_true | y_pred")
    print("-------------------")

    for number, (true, predicted) in enumerate(
        zip(y_true, y_pred, strict=True),
        start=1,
    ):
        print(f"{number:2} | {true:6} | {predicted:6}")

    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)


if __name__ == "__main__":
    main()
