from src.preprocess import load_lfw, flatten_images, split_data
from src.pca_model import load_pca
from src.classifier import load_clf, evaluate

if __name__ == '__main__':
    X, y, target_names, img_shape = load_lfw(min_faces_per_person=50, resize=0.4)
    X_flat = flatten_images(X)
    _, X_test, _, y_test = split_data(X_flat, y, test_size=0.25)

    pca = load_pca('models/pca.joblib')
    clf = load_clf('models/clf.joblib')

    X_test_pca = pca.transform(X_test)
    y_pred = evaluate(clf, X_test_pca, y_test, target_names)
