.. _miscellaneous:

Miscellaneous
=============

.. _nbody_simulation_utilities:

N-body simulation utilities
---------------------------
``celmech`` provides a number of utility functions for working with REBOUND simulations. These include:
    1. :func:`get_simarchive_integration_results<celmech.nbody_simulation_utilities.get_simarchive_integration_results>`
       reads the data stored in a `rebound` `SimulationArchive`_ as a dictionary
    2. :func:`set_time_step<celmech.nbody_simulation_utilities.set_time_step>`
       sets the time step of a simulation to a user-specified fraction of the
       shortest perihelion passage timescale.
    3. :func:`calculate_mutual_inclinations<celmech.nbody_simulation_utilities.calculate_mutual_inclinations>` computes the mutual inclinations between pairs of planets in a simulation.
    4. :func:`align_simulation<celmech.nbody_simulation_utilities.align_simulation>` aligns a ``rebound`` simulation so that the total angular momentum is oriented along the 
       :math:`z` axis.


.. _other_misc:

Other
-----
Other useful odds and ends are gathered into the :mod:`celmech.miscellaneous` module.
These include:
        1. **Frequency analysis**: 
           The :func:`~celmech.miscellaneous.frequency_modified_fourier_transform` function
           provides a wrapper for the FMFT algorithm of `Šidlichovský & Nesvorný (1996)`_. 
           A jupyter notebook example is available `here`_.

        2. **AMD**: Functions related to angular momentum deficit (AMD) and 
           the analytic stability criteria (see `Laskar & Petit (2017)`_): 

           - :func:`~celmech.miscellaneous.compute_AMD`
           
           - :func:`~celmech.miscellaneous.AMD_stable_Q`
           
           - :func:`~celmech.miscellaneous.AMD_stability_coefficients`
           
           - :func:`~celmech.miscellaneous.critical_relative_AMD`

           - :func:`~celmech.miscellaneous.critical_relative_AMD_resonance_overlap`

        3. The function :func:`~celmech.miscellaneous.holman_weigert_stability_boundary` 
           which computes a stability criterion for circum-binary planets based on `Holman & Wiegert (1999)`_
        4. The functions :func:`~celmech.miscellaneous.sk` and :func:`~celmech.miscellaneous.Dsk`
           which give approximate disturbing function coefficient amplitudes as and their derivatives
           with respect to eccentricity. 
           See `Hadden & Lithwick (2018) <https://ui.adsabs.harvard.edu/abs/2018AJ....156...95H/abstract>`_
        5. Convenience functions :func:`~celmech.miscellaneous.get_symbol` and :func:`~celmech.miscellaneous.get_symbol0` for generating sympy symbols from LaTeX strings.
        
        6. The function :func:`~celmech.miscellaneous.truncated_expansion` for expanding and
           truncating sympy expressions after assigning variables (possibly different) orders
           in a book-keeping parameter.


.. _here: https://github.com/shadden/celmech/blob/style_update/jupyter_examples/FrequencyAnalysis.ipynb
.. _Laskar & Petit (2017): https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..72L/abstract
.. _Holman & Wiegert (1999): https://ui.adsabs.harvard.edu/abs/1999AJ....117..621H/abstract

N-body simulation utilities API
-------------------------------

.. automodule:: celmech.nbody_simulation_utilities
        :members: set_time_step,set_min_distance,get_simarchive_integration_results,get_canonical_heliocentric_orbits, add_canonical_heliocentric_elements_particle, align_simulation, calculate_mutual_inclinations

Other API
---------

.. automodule:: celmech.miscellaneous
    :members:

.. _SimulationArchive: https://github.com/hannorein/rebound/blob/main/ipython_examples/SimulationArchive.ipynb
