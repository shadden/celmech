.. _maps:

Symplectic Maps
===============

The ``celmech.maps`` module implements a couple of area-preserving maps,
:class:`~celmech.maps.StandardMap` and :class:`~celmech.maps.EncounterMap`.

Chirikov Standard Map
---------------------

The Chirikov standard map is a symplectic map that depends on a single
parameter, :math:`K`, defined by the equations

.. math::
        p' &= p + K  \sin\theta 
        \\
        \theta' &= \theta + p'~.

Comet Map
-------------

The comet map is a symplectic map that approximates the dynamics of particles on
highly eccentric orbits perturbed by a planet on a circular orbit.  It is
described in `Hadden & Tremaine (2024)
<https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.3054H/abstract>`_.

Encounter Map
-------------

The encounter map is a symplectic map that approximates the dynamics of a pair
of closely-spaced planets.  The map depends on three parameters ,
:math:`\epsilon,y`, and :math:`J`.  The map is defined by the equations

.. math::
        x' &= x + \epsilon f(\theta;y)
        \\
        \theta' &= \theta + 2\pi(J-x')

where 

.. math::
    f(\theta;y) = -\frac{4\pi}{3}\sum_{k}s_k(y)k\sin(k\theta)

and the :math:`s_k(y)` are the resonance amplitude functions given by the
:func:`~celmech.miscellaneous.sk` function.

API
---
.. automodule:: celmech.maps
    :members: StandardMap, CometMap, EncounterMap
