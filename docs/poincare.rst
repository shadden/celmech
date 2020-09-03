.. _poincare:

Poincare Module
===============

Introduction
------------

The Poincare module is used to represent planetary systems in terms of Poincare canonical variables.
These canonical variables are related to the Delauney elliptic variable,

.. math::
        \begin{eqnarray}
            \Lambda_i &=& \tilde{m}_i\sqrt{G{\tilde M}_ia_i}\nonumber\\
            \Gamma_i  &=& \tilde{m}_i\sqrt{G{\tilde M}_ia_i}(1-\sqrt{1-e_i^2})\nonumber\\
            Q_i &=& \tilde{m}_i\sqrt{{\tilde M}_ia_i}\sqrt{1-e_i^2}(1-\cos(I_i))
        \end{eqnarray}

where :math:`\mu_i\sim m_i` and  :math:`{\tilde m_i}\sim M_*`
the exact definitions of  :math:`\mu_i\sim m_i` and  :math:`{\tilde m_i}\sim M_*` depend on canonical Cartesian coordinate system from which the variables are derived (e.g., heliocentric, Jacobi, etc.).
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


API
---
.. autoclass:: celmech.hamiltonian.Hamiltonian
        :members:
        :special-members: __init__

.. automodule:: celmech.poincare
        :members:
        :special-members: __init__
