.. _install:

Installation
============

Requirements
------------

Installation of ``celmech`` will require a few additional Pyton packages. 

GitHub
------

In order to install ``celmech`` you will first need to clone the GitHub repository located `here <https://github.com/shadden/celmech>`_. This can be accomplished from the terminal by first navigating to the desired target directory and then issuing the command::

        git clone https://github.com/shadden/celmech.git

After you have cloned the git repository, navigate into the top celmech directory and issue the terminal command::
        
        python setup.py install

in order to install the Python package.

Pip
---
Coming soon!

.. _first_example:

A First Example
===============

Now that ``celmech`` is installed, we'll run through a short example of how to use it. In our example we'll use it in conjunction with the ``REBOUND`` N-body integrator to build and integrate a simple Hamiltonian model for the dynamics of a pair of planets in a mean motion resonance.


Setup
-----

We'll start by importing the requisite packages.

.. code:: python

        import numpy as np
        import rebound as rb
        from celmech import Poincare, PoincareHamiltonian
        from sympy import init_printing
        init_printing() # This will typeset symbolic expressions in LaTeX

Now we'll initialize a REBOUND simulation containing a pair of Earth-mass planets orbiting a central solar-mass star with a 3:2 period ratio commensurability.

.. code:: python

        sim = rb.Simulation()
        sim.add(m=1,hash='star')
        sim.add(m=3e-6,P = 1, e = 0.05)
        sim.add(m=3e-6,P = 3/2, e = 0.05,pomega = np.pi)
        sim.move_to_com()

We'd like a construct a Hamiltonian that can capture the dynamical evolution of the system. 

.. code:: python
        
        pvars = Poincare.from_Simulation(sim)
        pham = PoincareHamiltonian()

We've now generated a Hamiltonian model  using the ``PoincareHamiltonian`` class. We can examine the symbolic expression for our Hamiltonian, stored as 

.. code:: python

        pham.H

which should display:

.. math::

        - \frac{G^{2} M_{2}^{2} m_{2}^{3}}{2 \Lambda_{2}^{2}} - \frac{G^{2} M_{1}^{2} m_{1}^{3}}{2 \Lambda_{1}^{2}}

This expression is the Hamiltonian of two non-interacting Keplerian orbits expressed in canonical variables used by ``celmech``.
The canonical momenta for the :math:`i`-th planet are defined[#]_ in terms of the planet's standard `orbital elements <https://en.wikipedia.org/wiki/Orbital_elements>`_ :math:`(a_i,e_i,I_i,\lambda_i,\varpi_i,\Omega_i)` and mass parameters :math:`\mu_i\sim m_i` and :math:`M_i \sim M_*`:

.. math::
        
        \Lambda_i = \mu_i \sqrt{G M_i a_i}\\
        \kappa_i = \sqrt{2\Lambda_i(1-\sqrt{1-e_i^2})}\cos\varpi_i\\
        \sigma_i = \sqrt{2\Lambda_i\sqrt{1-e_i^2}(1-\cos I_i)}\cos\Omega_i

and their conjugate coordinates are:

.. math::
        \lambda_i \\
        \eta_i = -\kappa_i\tan\varpi_i \\
        \rho_i = -\sigma_i\tan\Omega_i 

This is the default behavior of the ``PoincareHamiltonian`` class: upon initialization, it will only contain the terms corresponding to Keplerian orbits of the particles. Next, we will add terms to model the effect of the 3:2 mean motion resonance. This can be done conveniently with 

.. code:: python

        pham.add_all_MMR_and_secular_terms(3,1,1)
        pham.H

which should now display

.. math::

        - \frac{C^{0,0,0,0;(1,2)}_{0,0,0,0,0,0} G^{2} M_{2}^{2} m_{1}}{\Lambda_{2}^{2} M_{1}} m_{2}^{3} - \frac{C^{0,0,0,0;(1,2)}_{3,-2,-1,0,0,0} G^{2} M_{2}^{2} m_{1}}{\Lambda_{2}^{2} M_{1}} m_{2}^{3} \left(\frac{\eta_{1}}{\sqrt{\Lambda_{1}}} \sin{\left (2 \lambda_{1} - 3 \lambda_{2} \right )} + \frac{\kappa_{1}}{\sqrt{\Lambda_{1}}} \cos{\left (2 \lambda_{1} - 3 \lambda_{2} \right )}\right) - \frac{C^{0,0,0,0;(1,2)}_{3,-2,0,-1,0,0} G^{2} M_{2}^{2} m_{1}}{\Lambda_{2}^{2} M_{1}} m_{2}^{3} \left(\frac{\eta_{2}}{\sqrt{\Lambda_{2}}} \sin{\left (2 \lambda_{1} - 3 \lambda_{2} \right )} + \frac{\kappa_{2}}{\sqrt{\Lambda_{2}}} \cos{\left (2 \lambda_{1} - 3 \lambda_{2} \right )}\right) - \frac{G^{2} M_{2}^{2} m_{2}^{3}}{2 \Lambda_{2}^{2}} - \frac{G^{2} M_{1}^{2} m_{1}^{3}}{2 \Lambda_{1}^{2}}

A call to ``pham.add_all_MMR_and_secular_terms(p,q,order)`` adds all disturbing function terms associated with the :math:`p:p-q` resonance up to order ``order`` in eccentricities and inclinations, along with all secular terms up to the same order. In our case, we have added the two first-order cosine terms associated with the 3:2 MMR. 

In addition to storing a purely symbolic expression for 

Integration
-----------

Now that we have a Hamiltonain model, we'll integrate it

.. [#] The precise definitions of the orbital elements and mass parameters :math:`\mu_i,M_i` depend on the adopted coordinate system.  By default ``celmech`` uses canonical heliocentric coordinates.  
