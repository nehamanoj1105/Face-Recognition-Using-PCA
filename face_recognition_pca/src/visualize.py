import matplotlib.pyplot as plt
from src.preprocess import load_lfw
from src.pca_model import load_pca

def plot_eigenfaces(pca, img_shape, n_row=3, n_col=5, out='examples/eigenfaces.png'):
    eigenfaces = pca.components_.reshape((pca.n_components_,) + img_shape)
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(eigenfaces[i], cmap='gray')
        plt.xticks(())
        plt.yticks(())
        plt.title(f'PC {i+1}')
    plt.tight_layout()
    plt.savefig(out)
    print('Saved', out)

if __name__ == '__main__':
    X, y, target_names, img_shape = load_lfw(min_faces_per_person=50, resize=0.4)
    pca = load_pca('models/pca.joblib')
    plot_eigenfaces(pca, img_shape)
