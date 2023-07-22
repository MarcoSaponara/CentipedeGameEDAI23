import numpy as np
        

class KStrategy():
    def __init__(self,
                 kernel : np.ndarray,
                ):
        
        self.kernel = kernel
        assert kernel.shape[0] == kernel.shape[1]
        assert np.allclose(kernel.sum(axis=1), 1.)
        
    def calculate_mixed_strategy(self, 
                                 k = None,
                                 start = None,
                                ) -> np.ndarray:
        if k is None:
            k : int = 0
        else:
            assert k >= 0
            
        l = self.kernel.shape[0]
        
        if start is None:
            start = 2 * np.ones(l, dtype = float)/l
        else:
            assert len(start) == l
            assert np.isclose(start[:l//2].sum(),1.0)
            assert np.isclose(start[l//2:].sum(),1.0)
   
        return start @ np.linalg.matrix_power(self.kernel, k)


    def calculate_strategy_for_infinite_k(self,
                                          k_test = 1000,
                                          tol = 1.0e-8,
                                         ):
        eigenvalues, eigenvectors = eig(self.kernel, left = True, right = False)
        sd = abs(eigenvectors[:, np.argmax(eigenvalues.real)].real)
        sd /= sd.sum()
        
        test = np.linalg.matrix_power(self.kernel, k_test)[0,:]
        assert np.isclose(np.abs(sd - test).sum(), tol)
        
        return 2 * sd
        
        