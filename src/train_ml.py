from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from feature_extraction import load_images_and_labels, extract_features


def train_model():
    print("📦 Loading data...")
    images, labels = load_images_and_labels("data")

    print("🔍 Extracting features...")
    features = extract_features(images)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    print("🌳 Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("✅ Accuracy:", acc)

    # ✅ SAVE MODEL HERE (INSIDE FUNCTION)
    joblib.dump(clf, "model_rf.pkl")
    print("✅ ML model saved!")

    return clf


if __name__ == "__main__":
    train_model()