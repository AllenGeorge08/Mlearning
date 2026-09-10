from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml 
from sklearn.decomposition import PCA as SklearnPCA


mnist = fetch_openml("mnist_784",version=1,as_frame=False,parser="auto")
X_mnist = mnist.data[:5000].astype(float)
y_mnist = mnist.target[:5000].astype(int)

pca = SklearnPCA(n_components=2)
X_sklearn_pca = pca.fit_transform(X_mnist)


# print(f"\nOur PCA explained variance:     {pca_2d.expl/ained_variance_ratio_}")
print(f"Sklearn PCA explained variance: {pca.explained_variance_ratio_}")


tsne = TSNE(
    n_components=2, #784 dim  -> 2 dim
    perplexity=30, #how large of a neighbourhood to consider
    random_state=42  #random seed - to reproduce the result
)
X_tsne = tsne.fit_transform(X_mnist)
print(f"\nt-SNE output shape: {X_tsne.shape}")

# X_tsne = tsne.fit_transform(X_mnist)

try:
    from umap import UMAP
    reducer = UMAP(n_components=2,n_neighbors=15,min_dist=0.1,random_state=42)
    X_umap = reducer.fit_transform(X_mnist)
    print(f"UMAP output shape : {X_umap.shape}")
except ImportError:
    print("Install umap-learn: pip install umpa-learn")