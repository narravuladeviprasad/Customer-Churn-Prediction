import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# 1. Load the dataset
df = pd.read_csv("Churn_Modelling.csv")

# 2. Drop unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 3. Encode categorical columns
# Note: Spain is encoded as 2. France=0, Germany=1.
# Note: Male is encoded as 1. Female=0.
le_geo = LabelEncoder()
le_gender = LabelEncoder()
df['Geography'] = le_geo.fit_transform(df['Geography'])
df['Gender'] = le_gender.fit_transform(df['Gender'])

# --- Exploratory Data Analysis (EDA) with Graphs ---
print("--- Exploratory Data Analysis ---")

# Graph 1: Churn Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Exited', data=df)
plt.title('Distribution of Churned vs. Retained Customers')
plt.xlabel('Exited (0: No, 1: Yes)')
plt.ylabel('Number of Customers')
plt.show()

# Graph 2: Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# 4. Define features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# 5. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 7. Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# 8. Train, evaluate, and visualize each model
results = {}
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for ROC curve

    # Store results for final comparison plot
    results[name] = {'y_pred': y_pred, 'y_pred_proba': y_pred_proba}

    # Print Classification Report
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Graph 3: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Exited', 'Exited'],
                yticklabels=['Not Exited', 'Exited'])
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# --- Model Comparison Graphs ---

# Graph 4: Combined ROC Curves
plt.figure(figsize=(10, 8))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
    auc = roc_auc_score(y_test, res['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.50)') # Dashed diagonal
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# Graph 5: Feature Importance for Tree-Based Models
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        plt.figure(figsize=(10, 6))
        importances.nlargest(10).plot(kind='barh')
        plt.title(f'Top 10 Feature Importances for {name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis() # Display feature with highest importance at the top
        plt.show()
