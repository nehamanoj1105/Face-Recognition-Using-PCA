import argparse
import os
from src.preprocess import load_lfw, flatten_images, split_data
from src.pca_model import train_pca, save_pca
from src.classifier import train_clf, save_clf

def main(args):
    X, y, target_names, img_shape = load_lfw(min_faces_per_person=args.min_faces, resize=args.resize)
    X_flat = flatten_images(X)
    X_train, X_test, y_train, y_test = split_data(X_flat, y, test_size=args.test_size)

    print('Training PCA...')
    pca = train_pca(X_train, n_components=args.n_components, whiten=True)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    print('Training classifier...')
    clf = train_clf(X_train_pca, y_train)

    os.makedirs('models', exist_ok=True)
    save_pca(pca, 'models/pca.joblib')
    save_clf(clf, 'models/clf.joblib')
    print('Models saved to models/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_components', type=int, default=150)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--min_faces', type=int, default=50)
    parser.add_argument('--resize', type=float, default=0.4)
    args = parser.parse_args()
    main(args)
