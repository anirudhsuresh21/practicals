import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier

# Load diabetes dataset
df = pd.read_csv('SAheart.csv')
print(df.head(10))
print(df.info())
df.drop(columns=['famhist'], inplace=True)  # Drop 'famhist' column

# Prepare features and target
X = df.drop('chd', axis=1)
y = df['chd']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create and train decision trees
clf = DecisionTreeClassifier(max_depth=1, random_state=22)
clf2 = DecisionTreeClassifier(random_state=22)
clf = clf.fit(X_train, y_train)
clf2 = clf2.fit(X_train, y_train)

print("\nMetrics")
y_pred = clf.predict(X_test)
clf_accu = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {clf_accu * 100:.2f}%")

# Bagging
estimator_range = [4,8,12,24,36]
scores = []
models = []
for n_estimators in estimator_range:
    bg = BaggingClassifier(estimator=clf2, n_estimators=n_estimators, random_state=22)
    bg.fit(X_train, y_train)
    models.append(bg)
    scores.append(accuracy_score(y_true=y_test, y_pred=bg.predict(X_test)))

plt.figure(figsize=(9,6))
plt.plot(estimator_range, scores)
plt.xlabel("n_estimators", fontsize=18)
plt.ylabel("score", fontsize=18)
plt.tick_params(labelsize=16)
plt.show()
print(f"\nAccuracy of 5th Bagging Model: {scores[4] * 100:.2f}%")

# Bootstrapping
accuracy = []
n_iterations = 100
for i in range(n_iterations):
    X_bs, y_bs = resample(X_test, y_test, replace=True)
    y_hat = models[4].predict(X_bs)
    score = accuracy_score(y_bs, y_hat)
    accuracy.append(score)

sns.kdeplot(accuracy)
plt.title("Accuracy across 100 bootstrap samples of the held-out test set")
plt.xlabel("Accuracy")
plt.show()

# Boosting
ada = AdaBoostClassifier(estimator=clf, n_estimators=50, learning_rate=1.0, random_state=22)
ada.fit(X_train, y_train)
ada_accuracy = accuracy_score(y_test, ada.predict(X_test))
print(f'\nAccuracy of the weak learner (Decision Tree): {clf_accu * 100:.2f}%')
print(f'Accuracy of AdaBoost model: {ada_accuracy * 100:.2f}%')

# Cross-Validation
cv_scores = cross_val_score(clf2, X_train, y_train, cv=5, scoring='accuracy')
print(f'\nCross-Validation Results (Accuracy): {cv_scores}')
print(f'Mean Accuracy: {cv_scores.mean()}\n')
