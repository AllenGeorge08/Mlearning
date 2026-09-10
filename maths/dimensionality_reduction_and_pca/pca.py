import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components
        self.components = None  
        self.mean = None 
        self.eigenvalues = None 
        self.explained_variance_ratio = None 

    
    def fit(self,X):
        # Center
        self.mean = np.mean(X,axis=0)
        X_centered = X-self.mean 

        # 2. Covariance 
        cov_matrix = np.cov(
            X_centered,
            rowvar=False
        )

        # Eigendecomposition
        eigenvalues,eigenvectors = np.linalg.eigh(
            cov_matrix 
        )
        
        #Sort largest -> smallest 
        sorted_idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:,sorted_idx]

        # Keep top k
        self.components = (
            eigenvectors[:,:self.n_components].T 
        )

        self.eigenvalues = (
            eigenvalues[:self.n_components]
        )

        # Explained variance 
        total_var = np.sum(eigenvalues)

        self.explained_variance_ratio = (
            self.eigenvalues/total_var 
        )

        return self 

    
    def transform(self,X):
        # Center using training mean
        X_centered = X-self.mean 
        #Project onto principal components 
        return X_centered@self.components.T 
    

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)





pca = PCA(n_components=2)
X = np.array([[2,3,4],[5,4,8]])
X_reduced = pca.fit_transform(X)
print(f"X Reduced : {X_reduced}")