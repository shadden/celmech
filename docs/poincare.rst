.. _poincare:

Poincare Module
===============

.. _poincare_intro:

Introduction
------------

The Poincare module is used to represent planetary systems in terms of Poincare canonical variables.
These canonical variables are related to the Delauney elliptic variable,

.. math::
        \begin{eqnarray}
            \Lambda_i &=& \mu_i\sqrt{G{ M}_ia_i}\nonumber\\
            \Gamma_i  &=& \mu_i\sqrt{G{ M}_ia_i}(1-\sqrt{1-e_i^2})\nonumber\\
            Q_i &=& \mu_i\sqrt{{M}_ia_i}\sqrt{1-e_i^2}(1-\cos(I_i))
        \end{eqnarray}

where :math:`\mu_i\sim m_i` and  :math:`{\tilde m_i}\sim M_*`
the exact definitions of  :math:`\mu_i\sim m_i` and  :math:`{\tilde m_i}\sim M_*` depend on canonical Cartesian coordinate system from which the variables are derived (e.g., heliocentric, Jacobi, etc.).
``celmech`` uses :ref:`canonical heliocentric coordinates <ch_coordinates>`, described in greater detail below.
The associated canonical conjugate coordinates 
are the mean longitudes :math:`\lambda_i, \gamma_i=-\varpi_i, 
\mathrm{ and } q_i=-\Omega_i` where 
:math:`\varpi_i` is the longitude of periapse and 
:math:`\Omega_i` is the longitude of ascending node.
Instead of the variables :math:`(q_i,Q_i) \mathrm{ and } (\gamma_i,\Gamma_i)`, ``celmech`` formulates equations of motion in terms of 'cartesian-style' canonical coordinate-momentum pairs

.. math::
        \begin{eqnarray}
        (\eta_i,\kappa_i)&=&\sqrt{2\Gamma_i} \times (\sin\gamma_i,\cos\gamma_i)\\
        (\sigma_i,\rho_i)&=&\sqrt{2Q_i} \times (\sin q_i,\cos q_i)
        \end{eqnarray}

which have the advantage of being well-defined for :math:`e_i\rightarrow 0` and :math:`I_i\rightarrow 0`.


By default, ``celmech`` uses canonical heliocentric variables, in which the mass parameters are defined as

.. math::
        \tilde{m}_i = \frac{m_iM_*}{M_* + m_i}\\
        \tilde{M}_i = {M_* + m_i}

.. _ch_coordinates:

Canonical Heliocentric Coordinates
----------------------------------

``celmech`` uses canonical heliocentric coordinates in order to formulate the equations of motion.

Cartesian Heliocentric Coordinates
**********************************

If :math:`\pmb{u}_i` are the Cartesian position vectors of 
an :math:`N`-body system in an inertial frame 
and :math:`\tilde{\pmb{u}}_i=m_i\frac{d}{dt}\pmb{u}_i` are their corresponding conjugate momenta,
the canonical heliocentric coordinates :math:`\pmb{r}_i` are given by

.. math::
        \begin{pmatrix}
               \pmb{r}_0\\
               \pmb{r}_1\\
               \pmb{r}_2\\
               \vdots \\
               \pmb{r}_{N-1}
        \end{pmatrix}
        =
        \begin{pmatrix}
           1 &  0 &  0 &  \cdots & 0 \\
           -1 & 1 &  0 &  \cdots & 0 \\
           -1 & 0 &  1 & \cdots & 0 \\
           \vdots & \vdots & \vdots & \ddots & 0\\
           -1 & 0 &  0 & \cdots & 1 
           \end{pmatrix}
           \cdot
        \begin{pmatrix}
               \pmb{u}_0\\
               \pmb{u}_1\\
               \pmb{u}_2\\
               \vdots \\
               \pmb{u}_{N-1}
        \end{pmatrix}

and their conjugate momenta are

.. math::
        \begin{pmatrix}
              \tilde{\pmb{r}}_0\\
              \tilde{\pmb{r}}_1\\
              \tilde{\pmb{r}}_2\\
               \vdots \\
               \tilde{\pmb{r}}_{N-1}
        \end{pmatrix}
        =
        \begin{pmatrix}
           1 &  1 &  1 &  \cdots & 1 \\
           0 & 1 &  0 &  \cdots & 0 \\
           0 & 0 &  1 & \cdots & 0 \\
           \vdots & \vdots & \vdots & \ddots & 0\\
           0 & 0 &  0 & \cdots & 1 
           \end{pmatrix}
           \cdot
        \begin{pmatrix}
              \tilde{\pmb{u}}_0\\
              \tilde{\pmb{u}}_1\\
              \tilde{\pmb{u}}_2\\
               \vdots \\
               \tilde{\pmb{u}}_{N-1}
        \end{pmatrix}

The :math:`N`-body Hamiltonian is independent of :math:`\pmb{r}_0` and,
consequently :math:`\tilde{\pmb{r}}_0`, i.e., the total momentum of the system, is conserved.
In the new coordinates, the Hamiltonian becomes

.. math::
        \begin{eqnarray}
        H &=& \sum_{i=1}^{N-1}H_{\mathrm{kep},i} + \sum_{i=1}^{N-1} \sum_{j=1}^{i} H_{\mathrm{int.}}^{(i,j)} \\
        H_{\mathrm{kep},i} &=& \frac{1}{2}\frac{\tilde{r}_i^2}{\mu_i}   - \frac{GM_i\mu_i}{r_i} \\
        H_{\mathrm{int.}}^{(i,j)} &=& -\frac{Gm_im_j}{|\pmb{r}_i - \pmb{r}_j|} + \frac{\tilde{\pmb{r}}_i \cdot \tilde{\pmb{r}}_j }{m_0}
        \end{eqnarray}

Above, each :math:`H_{\mathrm{kep},i}` is the Hamiltonian of a two-body probelm with reduced mass :math:`\mu_i`.
We can therefore rewrite it in terms of the canonical Delauney variables defined :ref:`above <poincare_intro>` as

.. math::
        H_{\mathrm{kep},i} = -\frac{G^2M_i^2\mu^3}{2\Lambda_i^2}


Heliocentric Orbital Elements
*****************************

The Delauney variables, 
derived via a canonical transformation from the canonical heliocentric coordinates,
are defined in terms of a set of orbital elements.
A set of six `Keplerian orbital elements <https://en.wikipedia.org/wiki/Orbital_elements>`_
, :math:`(a_i,e_i,I_i,M_i,\varpi_i,\Omega_i)`, define by a mapping to Cartesian positions and velocities.
In the case of the elements appearing in the Delauney variables, the elements specify the
heliocentric position, :math:`\pmb{r}_i` and the
"velocity" :math:`\tilde{\pmb{r}}_i/\mu_i = \frac{M_* + m_i}{M_*}\frac{d \pmb{u}_i}{dt}`.
While this does not correspond to any physical velocity in the system, it ensures that 
the transformation from the coordinate-momentum pairs :math:`(\pmb{r},\tilde{\pmb{r}}_i)` 
to Delauney variables is canonical.

``celmech`` provides the 
functions :func:`nbody_simulation_utilities.get_canonical_heliocentric_orbits <celmech.nbody_simulation_utilities.get_canonical_heliocentric_orbits>` 
to compute these 'canonical heliocentric' elements from 
a REBOUND simulation along with 
:func:`nbody_simulation_utilities.add_canonical_heliocentric_elements_particle <celmech.nbody_simulation_utilities.add_canonical_heliocentric_elements_particle>` to add particles to a REOBOUND simulation by specifying the particles orbit in terms of these elements.




API
---
.. autoclass:: celmech.hamiltonian.Hamiltonian
        :members:
        :special-members: __init__

.. automodule:: celmech.poincare
        :members:
        :special-members: __init__