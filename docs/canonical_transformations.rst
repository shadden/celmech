.. _canonical_transformations:

Canonical Transformations
=========================

A transformation of canonical variables :math:`T:({q},{p}) \rightarrow
({q}',{p}')` that leaves Hamilton's equations unaltered so that 

.. math::
        \begin{align}
        \frac{d}{dt}q' &= \frac{\partial}{\partial p'} H(q(q',p'),p(q',p')) \\
        \frac{d}{dt}p' &= -\frac{\partial}{\partial q'} H(q(q',p'),p(q',p'))
        \end{align}

is called a canonical transformation. Often the equations of motion encountered
in celestial mechanics problems can be simplified by canonical coordinate
trasformations.

The `CanonicalTransformation` class
-----------------------------------

The class
:class:`~celmech.canonical_transformations.CanonicalTransformation`
provides a framework for composing and applying such transformations.
An instance of a
:class:`~celmech.canonical_transformations.CanonicalTransformation`
is initialized with a set of rules for transforming from old to new
variables and vice versa. These rules can be applied to any
expressions involving canonical variables by using the
:meth:`~celmech.canonical_transformations.CanonicalTransformation.old_to_new`
and
:meth:`~celmech.canonical_transformations.CanonicalTransformation.new_to_old`
methods. Additionally, the canonical transformation can be applied to
a :class:`~celmech.hamiltonian.Hamiltonian` instance in order to
produce a new Hamiltonian object that has the new variables as
canonical variables and with Hamiltonian appropriately transformed.

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

Next, we’ll build a canonical transformation to new variables
:math:`(Q_i,P_i)` according to :math:`\mathbf{Q} = T\cdot \mathbf{q}`
and :math:`\mathbf{P} = (T^{-1})^\mathrm{T} \cdot \mathbf{p}` for a
matrix :math:`T`. We’ll construct our transformations so that
:math:`Q_1 = 3q_1+2q_2`, the argument of the cosine term appearing in
the original Hamiltonian and we’ll set :math:`Q_2 = q_2`.

We use the :class:`~celmech.canonical_transformations.CanonicalTransformation`
class method
:method:`~celmech.canonical_transformations.CanonicalTransformation.from_linear_angle_transformation`
to produce a :class:`~celmech.canonical_transformations.CanonicalTransformation`
instance representing the desired transformation:

.. code:: ipython3

    Tmtrx = [[3,2],[0,1]]
    qp_old = ham.qp_vars
    ct = CanonicalTransformation.from_linear_angle_transformation(qp_old,Tmtrx)

We examine the resulting transformation by expressing each old canonical
variable in terms of new variables and vice versa

.. code:: ipython3

    for variable in ct.old_qp_vars:
        exprn = ct.old_to_new(variable)
        display((variable,exprn))
    
    for variable in ct.new_qp_vars:
        exprn = ct.new_to_old(variable)
        display((variable,exprn))


which produces the output

.. math::

    \displaystyle \left( q_{1}, \  \frac{Q_{1}}{3} - \frac{2 Q_{2}}{3}\right)



.. math::

    \displaystyle \left( q_{2}, \  Q_{2}\right)



.. math::

    \displaystyle \left( p_{1}, \  3 P_{1}\right)



.. math::

    \displaystyle \left( p_{2}, \  2 P_{1} + P_{2}\right)



.. math::

    \displaystyle \left( Q_{1}, \  3 q_{1} + 2 q_{2}\right)



.. math::

    \displaystyle \left( Q_{2}, \  q_{2}\right)



.. math::

    \displaystyle \left( P_{1}, \  \frac{p_{1}}{3}\right)



.. math::

    \displaystyle \left( P_{2}, \  - \frac{2 p_{1}}{3} + p_{2}\right)


Now we’ll use our canonical transformation to generate a new Hamiltonian
with the 
:meth:`~celmech.canonical_transformations.CanonicalTransformation.old_to_new_hamiltonian` method:

.. code:: ipython3

    kam = ct.old_to_new_hamiltonian(ham,do_reduction=True)
    kam.H


The final line displays the transformed Hamiltonian in terms of the new variables:

.. math::

    \displaystyle \frac{9 P_{1}^{2}}{2} + 2 P_{1} + P_{2} + \left(2 P_{1} + P_{2}\right)^{4} + \cos{\left(Q_{1} \right)}

After the transformation, the Hamiltonian does not depend on :math:`Q_2`, so the
corresponding momentum variable, :math:`P_2`, is conserved.  By passing the
keyword argument ``do_reduction=True``, we have elected to generate a new
Hamiltonian in terms of the reduced phase space variables, :math:`(Q_1,P_1)` in
which the conserved quantity :math:`P_2` appears as a parameter instead of a
dynamical variable. 

The reduced Hamiltonian still keeps track the full set of phase space
variables, :math:`(Q_1,Q_2,P_1,P_2)`, through the
:attr:`~celmech.hamiltonian.Hamiltonian.full_qp` attribute.
This stores a dictionary-like representation where the variable symbols
serve as keys and their numerical values as the associated value
entries.


It is particularly useful to have access to these values in situations
where one wishes to carry out an integration or other calculation in the
reduced phase space reduced Hamiltonian, but express results in the
original phase space variables using the inverse canonical
transformation. However, as we will see below, the numerical value
stored for :math:`Q_2` in ``full_qp`` will **NOT** be updated according
to :math:`\frac{d}{dt}Q_2 = \frac{\partial}{\partial P_2}` when the
system is integrated forward in time.

Common transformations
**********************


Reducing Degrees of Freedom
***************************

If the transformed Hamiltonian :math:`H' = H \circ T^{-1}` 
is independent of one or more of the new canonical cooridnate variables,
:math:`q'_i`, then the corresponding new momentum variable, :math:`p'_i`
is a constant of motion. Finding transformations that elminate the
dependence on one or more degrees of freedom allows one to consider a
simpler, lower-dimensional Hamiltonian problem involving only those
conjugate variable pairs, :math:`(q'_i,p'_i)`, for which both variables
appear explicitly in the Hamiltonian.

Lie Series Transformations
--------------------------

Constructing a Hamiltonian with a finite number of disturbing function terms
implicitly assumes that an infinite number of other terms can be ignored 
because they are rapdily oscillating such that there average effect on the
dynamics is negligible. In reality, these rapidly oscillating terms lead to
rapid oscillations in the dynamical variables under study. 

A generating function, :math:`\chi(q',p')`, is a function of canconical elements that is used to create a canonical transformation of variables. 
This is accomplished by means of a Lie transformation. 
The Lie transformation, :math:`f\rightarrow f'` of a function :math:`f`, induced by :math:`\chi`, is defined as

.. math::
        f'(q',p') = \exp[{\cal L}_\chi]f(q',p') = \sum_{n=0}^\infty \frac{1}{n!}{\cal L}_\chi^n f(q',p')

where :math:`{\cal L}_\chi= [\cdot,\chi]` is the Lie derivative with respect to :math:`\chi`, i.e., the Poisson bracket of a function with :math:`\chi`.

Generally, the goal of applying a Lie transformation is to eliminate the dependence of a Hamiltonian on a set of variables up to a specific order in some small parameter. 
In other words, usually one seeks to transform a Hamiltonian of the form :math:`H = H_0(p) + \epsilon H_1(q,p) + \epsilon^2H_2(q,p) + ...`, such that 

.. math::
        H'(q',p') = \exp[{\cal L}_\chi]H(q',p') = H'_0(p') + \epsilon^NH'_N(q',p')+....

where, in the new, transformed variables, :math:`(q',p')`, the Hamiltonian is integrable if one igonres terms of order :math:`\epsilon^N` and smaller.
In other words, :math:`p' = \exp[{\cal L}_{\chi(q,p)}]p` is a conserved quantity up to order :math:`\epsilon^{N-1}`.

The FirstOrderGeneratingFunction class
**************************************

``celmech`` provides the :class:`FirstOrderGeneratingFunction <celmech.generating_functions.FirstOrderGeneratingFunction>` class that can be used to apply transformations between osculating coordiantes used by :math:`N`-body simulatoins and transformed variables appropriate for that Hamiltonian models used by ``celmech``. 
These transformations will apply corrections at first order in planet-star mass ratio.

Generating function [DO SOME STUFF]

.. math:: 
        \begin{eqnarray}
        \bar{H}(\bar{\pmb{p}},\bar{\pmb{q}}) &=& \exp[L_{\chi_1}]{H}(\bar{\pmb{p}},\bar{\pmb{q}})\\
         &=&{H}_0(\bar{\pmb{p}}) + H_1(\bar{\pmb{p}},\bar{\pmb{q}}) + \{{H}_0(\bar{\pmb{p}}), {\chi_1} \} + {\cal O}(\epsilon^2)
         \end{eqnarray}


Choosing 

.. math:: \pmb{\omega}\cdot \nabla_{\pmb q} {\chi_1} = H_{1,\mathrm{osc}}(\bar{\pmb{p}},\bar{\pmb{q}})

eliminates oscliating terms to first order in :math:`\epsilon`.

To zeroth order in eccentricity, 

.. math:: H_{1,\mathrm{osc}} = -\frac{Gm_1m_2}{a_2}\left(\frac{1}{\sqrt{1 + \alpha^2 - 2\alpha\cos(\lambda_2 - \lambda_1})} - \frac{1}{2}b^{(0)}_{1/2}(\alpha) - \frac{\cos\psi}{\sqrt{\alpha}}\right)

Taking :math:`\chi_1 = \chi_1(\psi,\bar{\pmb{\Lambda}})` where :math:`\psi = \lambda_2-\lambda_1`,

.. math:: 
        \frac{\partial \chi_1}{\partial\psi} = -\frac{Gm_1m_2}{a_2\omega_\mathrm{syn}}f(\psi,\alpha)

with the solution 

.. math:: \chi_1 = -\frac{Gm_1m_2}{a_2\omega_\mathrm{syn}}\left({\frac{2}{1-\alpha}F\left(\frac{\psi}{2}\bigg| -\frac{4\alpha}{(1-\alpha)^2} \right)} - \frac{2\psi}{\pi} K(\alpha^2)- \frac{\sin\psi}{\sqrt{\alpha}}\right)

where :math:`K` and :math:`F` are complete and incomplete elliptic integrals of the first kind, respectively.

API
---

.. autoclass:: celmech.canonical_transformations.CanonicalTransformation
        :members:

.. autoclass:: celmech.generating_functions.FirstOrderGeneratingFunction
        :members:
