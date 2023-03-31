'''
STATE SELECTED DYNAMICS FOR N2+ e- -> N2- -> N2+ e- reaction.
Ref: Resonance-enhanced dissociation of a molecular ion below its excitation threshold
A. E. Orel, K. C. Kulander, Physical Review A, 54, 6, 1996

Soubhik M., Boston University, 
Boston MA, 02215
'''

import os
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from csaps import csaps
from FGHEVEN import *
from joblib import Parallel, delayed
import timeit
from alive_progress import alive_bar
import _rve_parameters

pathout = _rve_parameters.pathout
PECFILE = _rve_parameters.PECFILE


HBAR = 1.0
ANG_TO_BOHR = 1.8897259886
EV_TO_HARTREE = 1 / 27.21138602
AU_TO_FS = 0.02418884254
t_max = _rve_parameters.t_max #fs
t_max /= AU_TO_FS


def VARS():
    '''
    A function that returns dynamical parameters/constants for the simulation.

    Returns
    -------
    m_red : float
        Reduced mass of the nuclei (a.u.).
    m_e : float
        Mass of the electron (a.u.).
    dt : float
        Time interval in au
    N_tsteps : int
        Number of time steps for the simulation.
    '''

    m1= _rve_parameters.m_nuclei1 * 1836.15
    m2 = _rve_parameters.m_nuclei2 * 1836.15
    m_red = m1*m2/(m1+m2)
    m_e = 1.0 #mass of electron in a.u.
    dt = _rve_parameters.dt # fs
    dt /= AU_TO_FS  # au
    N_tsteps = int(t_max / dt)
    return m_red, m_e, dt, N_tsteps


def grid():
    '''
    A function that returns grid parameters for the simulation.

    Parameters
    ----------
    PECFILE : str
        Name of the Potential energy curve file.

    Returns
    -------
    x : np.ndarray
        The x-grid coordinates for 1D system in a.u.
    dx : float
        x Grid interval in a.u.
    k : np.ndarray
        The k-grid coordinates for 1D system in a.u.
    dk : float
        k Grid interval in a.u.
    N : int
        Number grid points in phase and momentum space.
    '''
    
    PECs = np.loadtxt(PECFILE)
    x_range = [PECs[:, 0].min(), PECs[:, 0].max()]
    N = _rve_parameters.NGrid
    x = np.linspace(x_range[0], x_range[1], N)
    dx = x[1]-x[0]

    # Grid in k-space
    dk = 2 * np.pi / (N * dx)
    k0 = -np.pi / dx
    k = k0 + dk * np.arange(N)

    return x, dx, k, dk, N


def potential_dict():
    '''
    A function that returns potential energy matrices in the grid.

    Parameters
    ----------
    PECFILE : str
        Name of the Potential energy curve file.
        
    Returns
    -------
    V_d : np.ndarray
        The resonance state potential in a.u.
    V_f : float
        The neutral state potential in a.u.
    W : np.ndarray
        The coupling potential in a.u.

    '''

    x, dx, k, dk, N = grid()
    PEC = np.loadtxt(PECFILE)
    potential_f = lambda x: csaps(PEC[:, 0], PEC[:, 1], x, smooth=0.99999)
    potential_d = lambda x: csaps(PEC[:, 0], PEC[:, 2], x, smooth=0.99994)
    Gamma = lambda x: csaps(PEC[:, 0], PEC[:, 3]*EV_TO_HARTREE, x, smooth=0.9994)

    W = np.array(np.sqrt(abs(Gamma(x))/(2.0*np.pi)), dtype=np.clongdouble)
    V_d = np.array( potential_d(x)-potential_f(x).min()- (0.5j *abs(Gamma(x))), dtype=np.clongdouble)
    V_f = np.array( potential_f(x)-potential_f(x).min(), dtype=np.clongdouble)

    return V_d, V_f, W


def psisolver():
    '''
    A function that returns 1D TISE solved wave-function using FGH method for 
    the neutral state of the moleucle.

    Parameters
    ----------
    PECFILE : str
        Name of the Potential energy curve file.
        
    Returns
    -------
    solver : <function psi:np.ndarray>
        1D TISE solved wave-function generator with psi, eigenval.

    '''

    x, dx, k, dk, N = grid()
    m, m_e, dt, N_tsteps = VARS()
    PEC_MRCISD = np.loadtxt(PECFILE)
    potential_i = lambda x: csaps(PEC_MRCISD[:, 0], PEC_MRCISD[:, 1], x, smooth=0.999999)

    print('Solving SE...')
    solver = FGHEVEN(x, potential_i(x)-potential_i(x).min(), m)
    print('SE solved.')
    return solver

def wf_norm(f, x):
    '''
    A function that defines regular norm for Hermitian Hamiltonian

    Parameters
    ----------
    f : np.ndarray
        Array to calculate norm for
    x : np.ndarray
        grid to integrate the array

    Returns
    -------
    float
        Norm of the array
    '''
    return integrate.simps(f.conj() * f, x)

def propagate(T_exp, V_exp, psi_x, x, dx, k, dk, dt, t, N, stride=1):
    '''
    A function that propagates a coupled system by Split-step Fourier method.

    Parameters
    ----------
    T_exp : np.ndarray
        Kinetic energy part of the propagator matrix 
        of dimension (2, 2) in exponential form for two states model coupled by W irreversibly.
    V_exp : np.ndarray
        Potential energy part of the propagator matrix 
        of dimension (2, 2) in exponential form
        
    psi_x: np.ndarray
        Wave function to propagate a full time step dt forwards
    x : np.ndarray
        The x-grid coordinates for 1D system in a.u.
    dx : float
        x Grid interval in a.u.
    k : np.ndarray
        The k-grid coordinates for 1D system in a.u.
    dk : float
        k Grid interval in a.u.
    N : int
        Number grid points in phase and momentum space.
    dt : float
        Time interval in a.u.
    t : float
        Total time propagated so far in dynamics 
    N: int
        length of the 1D phase-grid 
    stride: int
        How many forward stride to take in time-frame.

    Returns
    -------
    psi_x : np.ndarray
        Wave function propagated a full "stride" number step dt forwards
    psi_k : np.ndarray
        New wave in k-space
    t : float
        Updated time
    
    '''
    if stride > 0:
        # Half-step propagation in x-space
        psi_x = V_exp * psi_x

        # full-step propagation in k-space
        psi_mod_x = dx / np.sqrt(2 * np.pi) * np.exp(-1.0j * k[0] * x) * psi_x
        psi_mod_k = np.fft.fft(psi_mod_x)
        psi_k = psi_mod_k * np.exp(-1.0j * x[0] * dk * np.arange(N))
        psi_k *= T_exp
        psi_mod_k = psi_k * np.exp(1.0j * x[0] * dk * np.arange(N))
        psi_mod_x = np.fft.ifft(psi_mod_k)
        # Half-step propagation in x-space
        psi_x = np.sqrt(2 * np.pi) / dx * np.exp(1.0j * k[0] * x) * psi_mod_x

        psi_x = V_exp * psi_x
        t += stride * dt

    psi_mod_x = dx / np.sqrt(2 * np.pi) * np.exp(-1.0j * k[0] * x) * psi_x
    psi_mod_k = np.fft.fft(psi_mod_x)
    psi_k = psi_mod_k * np.exp(-1.0j * x[0] * dk * np.arange(N))

    return psi_x, psi_k, t



class get_sigma:

    '''
    A class that calculates RVE cross-section for a given set of kinetic energies
    of incoming electron.

    Attributes
    ----------
    ini_Ekin: np.ndarray
        Incoming electron kinetic energy in a.u.

    '''

    def __init__(self, ini_Ekin):
        '''
        Initializes the get_sigma class.

        Parameters
        ----------
        ini_Ekin: np.ndarray
            Incoming electron kinetic energy set in a.u.

        '''

        self.m, self.m_e, self.dt, self.N_tsteps= VARS()
        self.x, self.dx, self.k, self.dk, self.N = grid()
        self.V_d, self.V_f, self.W = potential_dict()
        self.ini_Ekin = ini_Ekin
        self.solver = psisolver()
        print ('Get_sigma is accessed (Should be done 1 time).')

    def psi_propagate_dict(self, Ientry):

        '''
        A function that propagates wave-function initially generated at t=0 in a given entry 
        channel of the RVE excitaion (v_i) with time.

        Parameters
        ----------
        Ientry : int
            Initial entry channel for the RVE dynamics
        
        Returns
        -------
        psi_dict_2D : np.ndarray
            A 2D array that stores wave-functions with each propagation time
            with a dimension of (Total time step * grid length)
        time_array : np.ndarray
            Time grid points of the simulation.
        '''

        T_exp = np.exp(-1.0j * self.dt * np.power(HBAR, 1.0) * np.power(self.k, 2.0) / (2.0 * self.m))
        V_exp = np.exp(-(1.0j * 0.5 * self.dt / HBAR) * (self.V_d))

        psi_d = np.array(self.solver.psi(Ientry), dtype=np.clongdouble)
        psi_d /= np.sqrt(wf_norm(psi_d, self.x))

        psi_x = psi_d * self.W
        t = 0.0
        time_array = np.array(0.0, dtype=np.longdouble)

        norm_d = []
        norm_d.append(np.real(wf_norm(psi_x, self.x)))

        psi_dict = np.empty([0], dtype=np.longdouble)
        psi_dict = np.append(psi_dict, psi_x)

        #width = 50
        with alive_bar(int(self.N_tsteps)-1, title='Storing psi dict progress') as bar:
            for i in range(1, int(self.N_tsteps)):
                psi_x, psi_k, t = propagate(T_exp, V_exp, psi_x, self.x, self.dx, self.k, self.dk, self.dt, t, self.N, 1)

                time_array = np.append(time_array, t)
                psi_dict = np.append(psi_dict, psi_x)
                norm_d.append(np.real(wf_norm(psi_x, self.x)))
                bar()


        psi_dict_2D = psi_dict.reshape((self.N_tsteps, self.N))
        print('Psi dictionary storing done.')
        return psi_dict_2D, time_array

    def sigma(self, psi_dict_2D, time_array, E_kin, Iexit):

        '''
        A function that calculates RVE cross-section for a given electron kinetic energy
        and exit channel

        Parameters
        ----------
        psi_dict_2D : np.ndarray
            A 2D array that owns wave-functions with each propagation time
            with a dimension of (Total time step * grid length)
        time_array : np.ndarray
            Time grid points of the simulation.
        E_kin: float
            Incoming electron kinetic energy in a.u.
        Iexit : int
            Exit channel for the RVE dynamics
        
        Returns
        -------
        E_kin : float
            Incoming electron kinetic energy in a.u.
        sigma : float
            RVE Cross-section in a.u.
        '''

        print('Ekin Sample No:%s, v_f: %s'% (self.ini_Ekin.index(E_kin), Iexit))

        chi_vf = np.array(self.solver.psi(Iexit), dtype=np.clongdouble)
        chi_vf /=np.sqrt(wf_norm(chi_vf, self.x))
        overlap = np.array(0.0, dtype=np.clongdouble)

        for i in range(1, int(self.N_tsteps)):
            overlap = np.append(overlap, np.exp(1.0j * (E_kin+2.0*self.solver.eigenval(0)) * time_array[i])
                            * integrate.simps((chi_vf.conj()* self.W* psi_dict_2D[i]), self.x) )

        T_matrix = (-1.0j/HBAR)* integrate.simps(overlap, time_array)
        sigma_value = ( (8.0*np.power(np.pi, 3)) /((2.0*E_kin*self.m_e)/HBAR/HBAR) ) *np.real(T_matrix * T_matrix.conj())
        sigma_value /=(ANG_TO_BOHR**2.0)
        return E_kin, sigma_value

def gen_sigma_joblib(nprocs, raw_Ekin, KE_vf):

    '''
        A function that parallalizes jobs over processors

        Parameters
        ----------
        nprocs : int
            No of processors
        E_kin: np.ndarray
            Incoming electron kinetic energy set in a.u.
        KE_vf : np.ndarray
            An array with inital electron KE and exit channels information
            in [(energy#1, channel no#1), (energy#2, channel no#1)..] format.
        
        Returns
        -------
        sigma_data: np.ndarray
            Contains initial electron energies and cross-sections, both in a.u.
        '''
    
    m, m_e, dt, N_tsteps = VARS()
    x, dx, k, dk, N = grid()

    fparam = open("".join(pathout+'RVE-PARAMETERS.out'), 'w')
    print('X min (bohr): %s \n'
          'X max (bohr): %s \n'
          'dt (fs): %s \n'
          'Time steps: %s \n'
          'Red. Mass (a.u.): %s \n'
          'Grid size: %s \n'%(x.min(), x.max(), dt*AU_TO_FS, N_tsteps, m, N), file=fparam)
    
    fparam.close()


    get_sigma_class = get_sigma(raw_Ekin)

    _psi_dict_2D, _time_array = get_sigma_class.psi_propagate_dict(0)

    sigma_data = Parallel(n_jobs=nprocs)(delayed(get_sigma_class.sigma)(_psi_dict_2D, _time_array, i, j) for i,j in KE_vf)
    return sigma_data





if __name__ == '__main__':

    print('Path of all outputs: %s' % pathout)

    start = timeit.default_timer()

    print('Time of Simulation %s fs'%int(t_max*AU_TO_FS))

    '''Create the initial Kinetic energy of electron: array'''
    n_ekin = _rve_parameters.nEkin
    max_ekin = _rve_parameters.Ekin_MAX*EV_TO_HARTREE
    min_ekin = _rve_parameters.Ekin_MIN*EV_TO_HARTREE

    step = float((max_ekin - min_ekin) / n_ekin)
    raw_Ekin = []
    for i in range(0, n_ekin):
        raw_Ekin.append(float(min_ekin + i * step))

    '''
    Create An array with inital electron KE and exit channels information
            in [(energy#1, channel no#1), (energy#2, channel no#1)..] format.
    '''
    KE_vf = []
    final_vf = np.arange(0, 10, 1)
    n_v_f = len(final_vf)

    for i in range(n_v_f):
        for j in range(len(raw_Ekin)):
            arg1 = raw_Ekin[j]
            arg2 = int(final_vf[i])
            KE_vf.append((arg1, arg2))

    '''Call the parallalized function to calculate the cross-section spectrum'''
    value = gen_sigma_joblib(8, raw_Ekin, KE_vf)


    '''Data analysis and plotting'''
    run_str = "".join(str(int(t_max*AU_TO_FS)) + 'fs')
    sigma_flat_array = np.empty(0)
    sigma_total = np.zeros(len(raw_Ekin))
    for i in range(len(raw_Ekin)*n_v_f):
        sigma_flat_array = np.append(sigma_flat_array, value[i][1])

    '''Cross section spectrum data printed to output files, images
        Initial entry channel is 0'''
    initial =0
    for i in range(n_v_f):
        sigma_total += sigma_flat_array[i*len(raw_Ekin):(i+1)*len(raw_Ekin):1]

        plt.figure(figsize=(6, 4))
        plt.plot((np.array(raw_Ekin))/EV_TO_HARTREE, sigma_flat_array[i*len(raw_Ekin):(i+1)*len(raw_Ekin):1],
                 '.-r',label = r'$v_i=%s \rightarrow v_f=%s$'%(initial, int(final_vf[i])))
        plt.legend(loc = 'upper right', fontsize = 12)
        plt.xlim(1.0, 5.0)
        plt.xlabel('Incident electron energy (eV)', fontsize = 14)

        image_path = os.path.join(pathout, 'N2_RVE_spectrum_%s-%s.png')
        image_name = image_path%(int(initial),int(final_vf[i]))
        plt.savefig(image_name, bbox_inches='tight')
        plt.close()

        image_path2 = os.path.join(pathout, 'N2_RVE_spectrum_%s-%s')
        fname = open( image_path2% (initial, int(final_vf[i])), 'w')

        for j in range(n_ekin):
            print("%12.10f\t%12.10f"%(value[j][0] / EV_TO_HARTREE,
                  sigma_flat_array[i * len(raw_Ekin):(i + 1) * len(raw_Ekin):1][j]), file=fname)
        fname.close()


    '''write to an output file: total cross-section: summing all channels'''
    kin_en = []
    sigma_e = []

    file1 = open(os.path.join(pathout, "N2_RVE_total_cross_%s_%s.dat") % (run_str, 'test'), "w")
    for i in range(len(raw_Ekin)):
        kin_en.append(raw_Ekin[i]/EV_TO_HARTREE)
        sigma_e.append(sigma_total[i])
        print("%15.12f \t %15.12f"%(value[i][0]/EV_TO_HARTREE, sigma_total[i]), file= file1)
    file1.close()


    stop = timeit.default_timer()
    print('Time elapsed!: ', stop - start)