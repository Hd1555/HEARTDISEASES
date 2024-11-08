from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("heart_disease.csv")

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Check target class balance
print("Target value counts before SMOTE:\n", data['target'].value_counts())

# If only one class is present, this print statement will alert you before applying SMOTE
if data['target'].nunique() == 1:
    print("Error: Only one class found in target. SMOTE cannot be applied.")
else:
    # Apply SMOTE to balance classes if there are multiple classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Class distribution after SMOTE:\n", pd.Series(y_resampled).value_counts())

    # Split resampled data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Initialize and train the RandomForest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)
    proba = rf_model.predict_proba(X_test)[:, 1]

    # Calculate accuracy and display results
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Cross-validation AUC
    cv_auc_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=5, scoring='roc_auc')
    print("Cross-Validation AUC Scores:", cv_auc_scores)
    print("Mean AUC Score:", cv_auc_scores.mean())

    # Calculate AUC only if there are two classes in y_test
    if len(y_test.unique()) > 1:
        roc_auc = roc_auc_score(y_test, proba)
        print("ROC AUC Score:", roc_auc)
    else:
        print("Only one class present in the test set. Skipping AUC calculation.")

    # Display confusion matrix
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature Importance Plot (horizontal and styled)
    feature_importances = rf_model.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title("Feature Importance in Random Forest Model")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Pair Plot for selected features
    sns.pairplot(data, vars=['age', 'thalach', 'chol', 'oldpeak'], hue='target', palette='coolwarm')
    plt.suptitle("Pair Plot of Selected Features by Target", y=1.02)
    plt.show()

    # Box Plot for each feature
    plt.figure(figsize=(12, 8))
    selected_features = ['age', 'thalach', 'chol', 'oldpeak']  # Choose a few features to visualize
    for i, feature in enumerate(selected_features, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='target', y=feature, data=data, palette='Set2')
        plt.title(f"Distribution of {feature} by Target")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
