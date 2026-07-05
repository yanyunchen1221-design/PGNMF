import numpy as np


class SimulatedMissingEvaluator:
    
    def __init__(self, mask_ratio=0.2, random_seed=42):
        self.mask_ratio = mask_ratio
        self.random_seed = random_seed
    
    def generate_simulated_missing_mask(self, X):
        np.random.seed(self.random_seed)
        n_cells, n_genes = X.shape
        mask = np.zeros((n_cells, n_genes), dtype=bool)
        
        for i in range(n_cells):
            nonzero_indices = np.where(X[i, :] > 0)[0]
            if len(nonzero_indices) > 0:
                n_to_mask = max(1, int(len(nonzero_indices) * self.mask_ratio))
                mask_indices = np.random.choice(nonzero_indices, n_to_mask, replace=False)
                mask[i, mask_indices] = True
        
        return mask
    
    def create_masked_data(self, X):
        mask = self.generate_simulated_missing_mask(X)
        X_masked = X.copy()
        X_masked[mask] = 0
        return X_masked