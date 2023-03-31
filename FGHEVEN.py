'''
This code is limited to even (N=2M) no of grid data points.
Original source in F90 form: 
https://formatika.de/index.jsp?content=directory&lib=products/Sources/formale%20Sprachen/Fortran/

Cite:  FGHEVEN
    F. Gogtas, G.G. Balint-Kurti and C.C. Marston,
    QCPE Program No. 647, (1993).

Soubhik M., Boston University, 
Boston MA, 02215
'''

import numpy as np
from scipy.linalg import eigh
hbar = 1
h = hbar*2.0*np.pi

class FGHEVEN:

    '''
    A class that utilizes Fourier Grid Hamiltonian method to solve Schrodinger equation.

    Attributes
    ----------
    x : np.ndarray
        The x-grid coordinates for 1D system.
    V : np.ndarray
        Potential energy in x grid.
    m: float
        Reduced mass of the nuclei.

    '''

    def __init__(self, x, V, m):
        '''
        Initializes the FGHEVEN class

        Parameters
        ----------
        x : np.ndarray
            The x-grid coordinates for 1D system.
        V : np.ndarray
            Potential energy in x grid.
        m: float
            Reduced mass of the nuclei.
        
        Returns
        -------
        eigval: np.ndarray
            eigenvalues of the Hamiltonian
        eigvec: np.ndarray
            eigen vectors of the Hamiltonian

        '''
        self.m = m
        self.x = x
        self.V = V
        self.dx = x[1]-x[0]
        self.N = len(self.x)
        self.L = self.N * self.dx
        
        self.T = np.zeros([self.N, self.N], dtype = np.clongdouble)
        
        for i in range (self.N):
            for j in range(self.N):
                if (i==j):
                    self.T[i,j] = (h**2.0)/(4.0*self.m*(self.L**2.0)) *((self.N-1.0)*(self.N-2)/6.0 + (self.N/2.0))
                else:
                    self.T[i,j] = (((-1)**(i-j)) / self.m) *( h/(2*self.L*np.sin(np.pi*(i-j)/self.N)))**2
                    
        self.V_diag= np.diag(self.V)
        self.H = self.T+ self.V_diag
        
        self.eigval, self.eigvec = eigh(self.H)
        
    def eigenval(self, v):
        '''
        A function that calculates eigenval for the 1D Hamiltonian

        Parameters
        ----------
        v: int
            Vibrational quantum number

        Returns
        -------
        eigval: float
            eigenvalue of the Hamiltonian for the vibrational quantum # v

        '''
        return self.eigval[v]
    
    def psi(self, v):
        '''
        A function that calculates eigenvectors for the 1D Hamiltonian

        Parameters
        ----------
        v: int
            Vibrational quantum number

        Returns
        -------
        eigval: float
            1D eigenvector of the Hamiltonian for the vibrational quantum # v

        '''
        return self.eigvec[:,v]