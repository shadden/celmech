.. _andoyer:

Andoyer Module
==============

Introduction
------------

The Andoyer module is used to represent the dynamical state of pair of planets in a standardized manner.
Using a series of canonical transformations and re-scalings, the Hamiltonian governing the dynamics of a
:math:`k`-th order MMR, for  :math:`k \le 3`, can (to leading order in eccentricity) be reduced the 'Andoyer Hamiltonian',

.. math::

        H_k(\Phi,\phi;\Phi') = 4\Phi^2 - 3\Phi'\Phi + \Phi^{k/2}\cos\phi

with the time measured in units of 

.. math::

        \tau \propto \left(\mu\right)^{k} t~.

This form of the Hamltonian...

API
---

.. automodule:: celmech.andoyer
        :members:
        :special-members: __init__
