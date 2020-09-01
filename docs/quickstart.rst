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

We've now generated a Hamiltonian model using the :class:`celmech.poincare.PoincareHamiltonian` class. We can examine the symbolic expression for our Hamiltonian, stored as 

.. code:: python

        pham.H

which should display:

.. math::

        - \frac{G^{2} M_{2}^{2} m_{2}^{3}}{2 \Lambda_{2}^{2}} - \frac{G^{2} M_{1}^{2} m_{1}^{3}}{2 \Lambda_{1}^{2}}

This expression is the Hamiltonian of two non-interacting Keplerian orbits expressed in canonical variables used by ``celmech``.
The canonical momenta for the :math:`i`-th planet are defined [#]_ in terms of the planet's standard `orbital elements <https://en.wikipedia.org/wiki/Orbital_elements>`_ :math:`(a_i,e_i,I_i,\lambda_i,\varpi_i,\Omega_i)` and mass parameters :math:`\mu_i\sim m_i` and :math:`M_i \sim M_*`:

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

Now that we have a Hamiltonain model, we'll integrate it and compare the results to direct :math:`N`-body.
First, we'll set up some preliminary python dictionaries and arrays to hold the results of both integrations.

.. code:: python

        # Here we define the times at which we'll get simulation outputs
        Nout = 150
        times = np.linspace(0 , 3e3, Nout) * sim.particles[1].P
        
        # These are the quantites we'll track in our rebound and celmech integrations
        keys = ['l1','l2','pomega1','pomega2','e1','e2','a1','a2'] 

        # These dictionaries will hold our results
        rebound_results= {key:np.zeros(Nout) for key in keys}
        celmech_results= {key:np.zeros(Nout) for key in keys}

        # These are the lists of particles in both simulations 
        # for which we'll save quantities.
        rb_particles = sim.particles
        cm_particles = pvars.particles


The :class:`celmech.PoincareHamiltonian` class inherits the method :meth:`celmech.hamiltonian.Hamiltonian.integrate` that can be used to evolve the system forward in much the same way as ``REBOUND``'s :meth:`rebound.Simulation.integrate` method.
Below is the main integration loop where we'll integrate our system and store the results: 

.. code:: python

        for i,t in enumerate(times):
            sim.integrate(t) # advance N-body
            pham.integrate(t) # advance celmech
            for j,p_rb,p_cm in zip([1,2],rb_particles[1:],cm_particles[1:]):
                # store N-body results
                rebound_results["l{}".format(j)][i] = p_rb.l
                rebound_results["pomega{}".format(j)][i] = p_rb.pomega
                rebound_results["e{}".format(j)][i] = p_rb.e
                rebound_results["a{}".format(j)][i] = p_rb.a

                # store celmech results
                celmech_results["l{}".format(j)][i] = p_cm.l
                celmech_results["pomega{}".format(j)][i] = p_cm.pomega
                celmech_results["e{}".format(j)][i] = p_cm.e
                celmech_results["a{}".format(j)][i] = p_cm.a

Finally, we'll plot the simulation results in order to compare them:

.. code:: python
        
        # First, we compute resonant angles for both sets of results
        for d in [celmech_results,rebound_results]:
            d['theta1'] = np.mod(3 * d['l2'] - 2 * d['l1'] - d['pomega1'],2*np.pi)
            d['theta2'] = np.mod(3 * d['l2'] - 2 * d['l1'] - d['pomega2'],2*np.pi)
        
        # Now we'll create a figure...
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(3,2,sharex = True,figsize = (12,8))
        for i,q in enumerate(['theta','e','a']):
            for j in range(2):
                key = "{:s}{:d}".format(q,j+1)
                ax[i,j].plot(times,rebound_results[key],'k.',label='$N$-body')
                ax[i,j].plot(times,celmech_results[key],'r.',label='celmech')
                ax[i,j].set_ylabel(key,fontsize=15)
                ax[i,j].legend(loc='upper left')

        #... and make it pretty
        ax[0,0].set_ylim(0,2*np.pi);
        ax[0,1].set_ylim(0,2*np.pi);
        ax[2,0].set_xlabel(r"$t/P_1$",fontsize=15);
        ax[2,1].set_xlabel(r"$t/P_1$",fontsize=15);
        
This should produce a figure that looks something like this:

.. image:: images/quickstart_example_plot.png
        :width: 600

Not too bad! Our ``celmech`` model reproduces the libration amplitudes and frequencies observed in the :math:`N`-body results quite successfully.

Next steps
----------

Check out ...

.. [#] The precise definitions of the orbital elements and mass parameters :math:`\mu_i,M_i` depend on the adopted coordinate system.  By default ``celmech`` uses canonical heliocentric coordinates.  
