import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Load data
data = pd.read_csv('rideshare_data_with_all_analysis.csv')

# Convert continuous target into categories (bins)
bin_edges = [0, 1, 2, 3, 4]
bin_labels = [0, 1, 2, 3]
data['surge_category'] = pd.cut(data['surge_multiplier'], bins=bin_edges, labels=bin_labels, include_lowest=True)

# Define target variable and a reduced set of features
target = 'surge_category'
features = ['price', 'distance', 'hour']  # Reduced feature set for simplicity
X = data[features]
y = data[target]

# Drop missing values
y = y.dropna()
X = X.loc[y.index]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Address class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train simpler models
models = {
    "Simple Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
    "Simple Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5, class_weight='balanced'),
    "Simple Random Forest": RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5, class_weight='balanced')
}

results = {}
for model_name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[model_name] = {
        "Accuracy": accuracy,
        "Classification Report": report
    }

# Display results
print("\n=== Model Performance ===")
for model_name, result in results.items():
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {result['Accuracy']:.2f}")
    print(f"Classification Report:\n{pd.DataFrame(result['Classification Report']).transpose()}")
