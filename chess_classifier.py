import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_and_preprocess_data(binary_class=True):
    """Load and preprocess data with option for binary classification"""
    try:
        df = pd.read_csv('data/games.csv')
    except FileNotFoundError:
        print("Error: Place games.csv in the data/ folder")
        exit()

    # Convert winner to binary (1=white wins, 0=black wins/draw)
    if binary_class:
        df = df[df['winner'].isin(['white', 'black'])]  # Remove draws
        df['target'] = (df['winner'] == 'white').astype(int)
    else:
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['winner'])

    # Feature engineering
    df['rating_diff'] = df['white_rating'] - df['black_rating']
    if 'moves' in df.columns:
        df['move_count'] = df['moves'].apply(lambda x: len(x.split()))
    
    features = ['rating_diff', 'turns', 'white_rating', 'black_rating']
    if 'move_count' in df.columns:
        features.append('move_count')

    # Drop rows with missing values
    df = df.dropna(subset=features + ['target'])
    
    return df[features], df['target']

def train_decision_tree(X_train, y_train):
    """Train and return Decision Tree model"""
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """Train and return SVM model"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def train_knn(X_train, y_train):
    """Train and return KNN model"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_model(model, X_test, y_test, model_name, scaler=None):
    """Evaluate model and save metrics"""
    if scaler:
        X_test = scaler.transform(X_test)
    
    y_pred = model.predict(X_test)
    
    # Save metrics
    with open(f'models/{model_name}_metrics.txt', 'w') as f:
        f.write(f"{model_name} Results\n")
        f.write("="*40 + "\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(y_test, y_pred)))
    
    # Plot confusion matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'models/{model_name}_confusion.png')
    plt.close()

def plot_decision_tree(model, feature_names):
    """Save decision tree visualization"""
    plt.figure(figsize=(20,10))
    plot_tree(model, 
              feature_names=feature_names, 
              class_names=['Black', 'White'], 
              filled=True, 
              rounded=True)
    plt.savefig('models/decision_tree.png')
    plt.close()

def main():
    print("Chess Game Binary Classifier")
    
    # Load data (binary classification)
    X, y = load_and_preprocess_data(binary_class=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    print("\nTraining Decision Tree...")
    dt_model = train_decision_tree(X_train, y_train)
    evaluate_model(dt_model, X_test, y_test, "decision_tree")
    plot_decision_tree(dt_model, X.columns.tolist())
    
    print("\nTraining SVM...")
    svm_model, svm_scaler = train_svm(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, "svm", svm_scaler)
    
    print("\nTraining KNN...")
    knn_model, knn_scaler = train_knn(X_train, y_train)
    evaluate_model(knn_model, X_test, y_test, "knn", knn_scaler)
    
    print("\nTraining complete! Check models/ folder for results.")

if __name__ == "__main__":
    main()