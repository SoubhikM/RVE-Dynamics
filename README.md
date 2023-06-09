## Resonance vibrational excitation (**RVE**) dynamics on complex potential energy surfaces.
### Case in hand: **$(^2\Pi_g)$** shape resonance of $N_2^-$

> RVE process looks like following when an incoming $e^-$ collides with a neutral $N_2$ molecule and the $^2\Pi_g$ resonance state is formed which subsequently decays with time and collapses to a vibrational channel $v_f$ of the neutral $N_2$ by emitting an electron.

```math
N_2(v_i=0)+e^-(E_{in}) \rightarrow N_2^-(^2\Pi_g)\rightarrow N_2(v_f)+e^-(E_{out})
```

> Energetically the RVE process looks like following if the neutral molecule is starting at lowest vibrational level ($v_i=0$).

```math
E_{in}\left(\frac{\hbar^2k_i^2}{2m_e}\right)+E_{N_2 (v_i=0)}=E_{out}\left(\frac{\hbar^2k_f^2}{2m_e}\right)+E_{N_2 (v_f)}=E
```

>> Where $v_i$ and $v_f$ are entry and exit channels respectively. $k_i$ and $k_f$ are incoming and outgoing momentum of electron. The energy conservation works as above, always mainataing a total energy of E.

> The cross-section spectrum is calculated from T-matrix defined as a Fourier transformed cross correlation function where the entry channel wave function ($v_i$=0), created on neutral surface, is excited to the resonance state and propagated by the non-Hermitian Hamiltonian $\hat{H_d} = \hat{T} + \hat{V} -i\frac{\Gamma}{2}$ comprising of nuclear kinetic energy operator, potential energy and the imaginary decay width respectively. The propagated wave-packet is projected on an exit channel $v_f$.

```math
\hat{T}_{v_f\leftarrow v_i}(E)=-\frac{i}{\hbar}\int_{0}^{\infty}e^{i{Et}/{\hbar}}\left \langle \phi_{v_f}| e^{-i \hat{H_d}t/\hbar}|\phi_{v_i}\right \rangle dE
```
> The cross-section spectrum $\sigma(E)$ subsequently is evaluated as follwoing

```math
\sigma_{v_f \leftarrow v_i}(E) = \frac{8\pi^3}{k_i^2} \left | \hat{T}_{v_f\leftarrow v_i}(E)\right|^2
```

> The following dependencies are needed to execute ``python3 N2_RVE.py``

        numpy 
        scipy
        matplotlib
        csaps
        joblib
        timeit
        alive_progress

> Execute the program

        python3 N2_RVE.py

> Change dynamical (*e.g.* time interval, grid point, time of simulation *etc.*) parameters in the following parameter file.

       _rve_parameters.py
       

## References
<a id="1">[1]</a> 
A. E. Orel, K. C. Kulander, 
Resonance-enhanced dissociation of a molecular ion below its excitation threshold, 
Physical Review A, 54, 6, 1996.

<a id="2">[2]</a> 
 L. Dub\'e, and A. Herzenberg,
Absolute cross sections from the "boomerang model" for resonant electron-molecule scattering,
Physical Review A, 20, 1, 194--213 (1979).
