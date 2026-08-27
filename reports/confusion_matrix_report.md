# Confusion Matrix Analysis

## 1. Objective

The objective of this exercise is to understand binary classification outcomes,
manually construct a confusion matrix, and explain common classification metrics.

## 2. Dataset

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

**Source:**
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

The dataset contains:

- 569 observations
- 30 numerical features
- Two classes: malignant and benign tumors

For this exercise, the classes are defined as:

- `1` — malignant tumor
- `0` — benign tumor

## 3. Model

The data was divided into training and testing subsets.

A `DecisionTreeClassifier` was trained using the following parameters:

```python
DecisionTreeClassifier(
    max_depth=2,
    random_state=42,
)
```

Ten observations from the test set were selected for manual analysis.

## 4. Meaning of Classification Outcomes

### True Positive (TP)

Actual class is positive, predicted class is positive. 

### True Negative (TN)

Actual class is negative, predicted class is negative.

### False Positive (FP)

Actual class is negative, predicted class is positive.

### False Negative (FN)

Actual class is positive, predicted class is negative. 

## 5. Predictions

| Observation | Actual class (`y_true`) | Predicted class (`y_pred`) | Outcome |
|---:|---:|---:|---|
| 1 | 1 | 1 | TP |
| 2 | 0 | 0 | TN |
| 3 | 1 | 1 | TP |
| 4 | 0 | 0 | TN |
| 5 | 0 | 1 | FP |
| 6 | 0 | 0 | TN |
| 7 | 0 | 0 | TN |
| 8 | 0 | 0 | TN |
| 9 | 0 | 0 | TN |
| 10| 0 | 0 | TN |

## 6. Manual Calculation

After reviewing all ten predictions:

- TP = 2
- TN = 7 
- FP = 1
- FN = 0

Verification:

```text
TP + TN + FP + FN = 10
```

## 7. Confusion Matrix

| | Predicted benign (`0`) | Predicted malignant (`1`) |
|---|---:|---:|
| Actual benign (`0`) | 7 | 1 |
| Actual malignant (`1`) | 0 | 2 |


## 8. Verification with scikit-learn


**Result:**

```text
[[7 1]
 [0 2]]
```

**Does it match the manual calculation?**

Yes

## 9. Written Questions

### 9.1 Why can accuracy be misleading?

Accuracy can be misleading on imbalanced sets. If there are two category and you algorithm choose one with bigger number of samples it can have a good accuracy but still it's a bad learning algorithm. 


### 9.2 What is the difference between precision and recall?

Precision describes relation of TP to all values that was marked as positive (TP+FP).

Precision = TP / (TP + FP)


And recall describes relation of TP is positive (TP+FN).

Recall = TP / (TP + FN)

### 9.3 When is reducing false negatives especially important?

Reducing false negatives can be specificly important in medicine. In this case we want to reduce as much as possible number of patience maked as negatives when they are positives for a certain desiase. 

### 9.4 When is reducing false positives especially important?

Reducing false positives is important when the consequences of a false positive
are more serious than those of a false negative. For example suppose that in machinery we mark some comptonent as positive if component if safe, and negative if it's defected. Then using such component can lead to major consequences. 

