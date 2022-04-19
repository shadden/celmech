.. _canonical_transformations:

Canonical Transformations
=========================

Introduction
------------

A transformation of canonical variables :math:`T:({q},{p}) \rightarrow
({q}',{p}')` that leaves Hamilton's equations unaltered so that 

.. math::
        \begin{align}
        \frac{d}{dt}q' &= \frac{\partial}{\partial p'} H(q(q',p'),p(q',p')) \\
        \frac{d}{dt}p' &= -\frac{\partial}{\partial q'} H(q(q',p'),p(q',p'))
        \end{align}

is called a canonical transformation. Often the equations of motion encountered
in celestial mechanics problems can be simplified by canonical coordinate
trasformations. The class
:class:`~celmech.canonical_transformations.CanonicalTransformation` provides a
framework for composing and applying such transformations. An instance of a
:class:`~celmech.canonical_transformations.CanonicalTransformation` is
initialized with a set of rules for transforming from old to new variables and
vice versa. These rules can be applied to any expressions involving canonical
variables by using the ``old_to_new`` and ``new_to_old`` methods. Additionally,
the canonical transformation can be applied to a
:class:`~celmech.hamiltonian.Hamiltonian` instance in order to produce a new
Hamiltonian object that has the new variables as canonical variables and with
Hamiltonian appropriately transformed.

As an example, we'll apply a canonical transformation to simplify the
toy Hamiltonian

.. math::
        H(q,p) = \frac{p_{1}^{2}}{2} + p_{2}^{4} + p_{2} + \cos{\left(3 q_{1} + 2 q_{2} \right)}

First, we'll set up a :class:`~celmech.hamiltonian.Hamiltonian` object to
represent this system

.. code:: python

        from celmech import CanonicalTransformation, Hamiltonian, PhaseSpaceState
        import sympy as sp
        import numpy as np
        p1,p2,q1,q2 = sp.symbols("p(1:3),q(1:3)")
        H = p1**2/2 + p2 +  p2**4 + sp.cos(3*q1 + 2 * q2)
        state=PhaseSpaceState([q1,q2,p1,p2],np.random.uniform(-0.5,0.5,4))
        ham = Hamiltonian(H,{},state)

Common transformations
---------------------------


Reducing Degrees of Freedom
---------------------------

If the transformed Hamiltonian :math:`H' = H \circ T^{-1}` 
is independent of one or more of the new canonical cooridnate variables,
:math:`q'_i`, then the corresponding new momentum variable, :math:`p'_i`
is a constant of motion. Finding transformations that elminate the
dependence on one or more degrees of freedom allows one to consider a
simpler, lower-dimensional Hamiltonian problem involving only those
conjugate variable pairs, :math:`(q'_i,p'_i)`, for which both variables
appear explicitly in the Hamiltonian.

API
---

.. autoclass:: celmech.canonical_transformations.CanonicalTransformation
        :members:
        :special-members: __init__

