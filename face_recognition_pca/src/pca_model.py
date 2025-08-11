import joblib
from sklearn.decomposition import PCA

def train_pca(X_train, n_components=150, whiten=True):
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=whiten)
    pca.fit(X_train)
    return pca

def save_pca(pca, path):
    joblib.dump(pca, path)

def load_pca(path):
    return joblib.load(path)
