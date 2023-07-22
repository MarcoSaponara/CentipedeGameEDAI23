import numpy as np

from scipy.linalg import circulant, eig
    
    
def eps_kernel_sym(nb_steps : int, 
                   gamma : float,
                   eps : float,
                  ):
    n : int = nb_steps//2 + 1 # nb_actions per player
    
    kernel = gamma * np.eye(2 * n, dtype = float)
    
    nbr_pl1 =  circulant(np.array([1. - eps] + (n - 1) * [eps/(n - 1)])).T
   
    nbr_pl2 = np.vstack((nbr_pl1[0,:], nbr_pl1[:-1,:]))
    
    kernel[n:,:n] = (1 - gamma) * nbr_pl1
    kernel[:n,n:] = (1 - gamma) * nbr_pl2
    
    return kernel


def eps_kernel_asyml(nb_steps : int, 
                     gamma : float,
                     eps : float,
                    ):
    n : int = nb_steps//2 + 1 # nb_actions per player
    
    kernel = gamma * np.eye(2 * n, dtype = float)
    
    nbr_pl1 = (1 - eps) * np.eye(n, dtype = float)
    nbr_pl1[0,0] = 1.
    for i in range(1, n):
        nbr_pl1[i,i-1] = eps 

    nbr_pl2 = np.vstack((nbr_pl1[0,:], nbr_pl1[:-1,:]))
    
    kernel[n:,:n] = (1 - gamma) * nbr_pl1
    kernel[:n,n:] = (1 - gamma) * nbr_pl2
    
    return kernel


def eps_kernel_asymr(nb_steps : int, 
                     gamma : float,
                     eps : float,
                    ):
    n : int = nb_steps//2 + 1 # nb_actions per player
    
    kernel = gamma * np.eye(2 * n, dtype = float)
    
    nbr_pl1 = (1 - eps) * np.eye(n, dtype = float)
    for i in range(n-1):
        nbr_pl1[i,i+1] = eps 
    nbr_pl1[-1,-1] = 1.
    
    nbr_pl2 = np.vstack((nbr_pl1[0,:], nbr_pl1[:-1,:]))
    
    kernel[n:,:n] = (1 - gamma) * nbr_pl1
    kernel[:n,n:] = (1 - gamma) * nbr_pl2
    
    return kernel


def qr_kernel(payoffs_pl1 : np.ndarray, 
              payoffs_pl2 : np.ndarray, 
              gamma : float,
              lam : float,
             ):
    assert len(payoffs_pl1) == len(payoffs_pl2)
    n : int = len(payoffs_pl1)//2 + 1 # nb_actions per player
    
    A = np.zeros((n, n), dtype = float)
    B = np.zeros((n, n), dtype = float)

    for i in range(n):
        for j in range(n):
            if j>=i:
                A[i,j] = payoffs_pl1[2*i]
                B[i,j] = payoffs_pl2[2*i]
            else:
                A[i,j] = payoffs_pl1[2*j + 1]
                B[i,j] = payoffs_pl2[2*j + 1]
                
    A = np.exp(lam * A.T)
    B = np.exp(lam * B)
    
    #normalize matrix by rows
    A /= A.sum(axis=1)[:, np.newaxis]
    B /= B.sum(axis=1)[:, np.newaxis]
    
    kernel = gamma * np.eye(2 * n, dtype = float)
    
    kernel[n:,:n] = (1 - gamma) * A
    kernel[:n,n:] = (1 - gamma) * B
    
    return kernel
