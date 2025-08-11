import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def train_clf(X_train_pca, y_train, C=1000.0, kernel='rbf'):
    clf = SVC(kernel=kernel, class_weight='balanced')
    clf.C = C
    clf.fit(X_train_pca, y_train)
    return clf

def save_clf(clf, path):
    joblib.dump(clf, path)

def load_clf(path):
    return joblib.load(path)

def evaluate(clf, X_test_pca, y_test, target_names):
    y_pred = clf.predict(X_test_pca)
    print(classification_report(y_test, y_pred, target_names=target_names))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    return y_pred
