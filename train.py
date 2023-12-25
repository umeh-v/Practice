import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def loadData(file_path):
    """
    Loads data from a CSV file.
    
    Args:
    file_path (str): The path to the CSV file.

    Returns:
    DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def preprocessData(X):
    """
    Preprocesses the dataset. Includes standard scaling.

    Args:
    X (DataFrame): Features.

    Returns:
    DataFrame: Preprocessed features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

def buildModel(X_train, y_train):
    """
    Builds and trains a Random Forest classifier.

    Args:
    X_train (DataFrame): Training features.
    y_train (Series): Training target variable.

    Returns:
    RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def assessModel(model, X_test, y_test):
    """
    Assesses the performance of the model.

    Args:
    model (RandomForestClassifier): The trained model.
    X_test (DataFrame): Testing features.
    y_test (Series): Testing target variable.

    Returns:
    float: The accuracy of the model.
    """
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

# Example usage
if __name__ == "__main__":
    # Load data
    data = loadData(r"C:\Users\user\Documents\AI_MSc\COM774\CW2\test_dataset.csv")  # Replace with the actual path to your dataset

    # Split the data into features and target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Preprocess data
    X_processed = preprocessData(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

    # Build and train the model
    model = buildModel(X_train, y_train)

    # Assess the model
    accuracy = assessModel(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy}")
