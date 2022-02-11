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
    4. :func:`algin_simulation` aligns a ``rebound`` simulation so that the total angular momentum is oriented along the 
       :math:`z` axis.


.. _other_misc:

Other
-----
Other useful odds and ends are gathered into the :mod:`celmech.miscellaneous` module.

N-body simulation utilities API
-------------------------------

.. automodule:: celmech.nbody_simulation_utilities
        :members: set_time_step,set_min_distance,get_simarchive_integration_results,get_canonical_heliocentric_orbits, add_canonical_heliocentric_elements_particle, align_simulation, calculate_mutual_inclinations

Other API
---------

.. automodule:: celmech.miscellaneous
    :members:

.. _SimulationArchive: https://github.com/hannorein/rebound/blob/main/ipython_examples/SimulationArchive.ipynb
