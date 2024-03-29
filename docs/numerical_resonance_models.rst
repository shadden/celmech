.. _numerical_resonance_models:

Numerical Resonance Models
==========================

Introduction
------------

Approximate equations of motion governing the dynamics of a mean motion
resonance can be generated by including terms from a distrubing function
expansion using the :class:`~celmech.poincare.PoincareHamiltonian` class.

However, ``celmech`` also allows users to derive equations governing mean
motion resonance dynamics without resorting to a truncated Taylor expansions in
eccentricities and inclinations.  `celmech` provides two such classes,
:class:`~celmech.numerical_resonance_models.PlanarResonanceEquations` and
:class:`~celmech.numerical_resonance_models.SpatialResonanceEquations`, are
described below.

Planar Resonance Equations
--------------------------
 
The :class:`~celmech.numerical_resonance_models.PlanarResonanceEquations` class
provides equations of motion governing a :math:`j\mathrm{:}j-k` mean-motion
resonance between two coplanar planets.  In order to do so, the class first
constructs a numerical Hamiltonian governing the system by means of numerical
integration.

Introduction
************

Starting from the full 3-body Hamiltonian governing a pair of planar planets, 

.. math::
        
        \begin{align}
                {\cal H}({\pmb \lambda},{\pmb \gamma};\pmb{\Lambda},\pmb{\Gamma}) &= H_\mathrm{Kep}(\pmb{\Lambda}) +  H_\mathrm{int}({\pmb \lambda},{\pmb \gamma};\pmb{\Lambda},\pmb{\Gamma})\\
                H_\mathrm{Kep}(\Lambda_i) &= -\sum_{i=1}^2\frac{G^2(M_*+m_i)^2\mu_i^3}{2\Lambda_i^2} \\
                H_\mathrm{int}(\lambda_i,\gamma_i;\Lambda_i,\Gamma_i) &= -\frac{Gm_1m_2}{|{\bf r}_1-{\bf r}_2|} + \frac{\tilde {\bf r}_1 \cdot \tilde {\bf r}_2}{M_*}~,
         \end{align}

the first step in constructing this Hamiltonian is to perform a canonical
transformation to new canonical coordinates

.. math::

        \begin{align}
        \begin{pmatrix}\sigma_1\\ \sigma_2 \\ Q \\ l \end{pmatrix}
        =
        \begin{pmatrix}
        -s & 1+s & 1 & 0 \\
        -s & 1+s & 0 & 1 \\
        -1/k & 1/k & 0 & 0 \\
        \frac{1}{2} & \frac{1}{2} & 0 & 0
        \end{pmatrix}
        \cdot
        \begin{pmatrix}\lambda_1 \\ \lambda_2 \\ \gamma_1 \\ \gamma_2 \end{pmatrix}
        \end{align}

where :math:`s = (j-k)/{k}`, along with new conjugate momenta 
defined implicitly in terms of the old momentum variables as

.. math::

        \begin{eqnarray}
        \Gamma_i &=& I_i \nonumber\\
        \Lambda_1 &=& \frac{L}{2} - P/k - s (I_1 + I_2) \nonumber \\
        \Lambda_2 &=& \frac{L}{2} + P/k + (1+s) (I_1 + I_2)~.\label{eq:equations_of_motion:momenta}
        \end{eqnarray}

Conservation of angular momentum implies that the quantity

.. math::
   L = \Lambda_1 + \Lambda_2 + \Gamma_1 + \Gamma_2~,
        
conjugate to the angle :math:`l`, is conserved

Next, we separate :math:`H_\mathrm{int}` into two pieces, 

.. math::
        H_\mathrm{int} &=  \bar{H}_\mathrm{int}(\sigma_i,I_i,P; L) + H_\mathrm{int,osc.}(\sigma_i,I_i,Q,P; L) \\
        \bar{H}_\mathrm{int}(\sigma_i,I_i,P; L) &= \frac{1}{2\pi}\int_{0}^{2\pi} H_\mathrm{int} dQ\\
        H_\mathrm{int,osc.} &= H_\mathrm{int} - \bar{H}_\mathrm{int}
        
The :class:`PlanarResonanceEquations
<celmech.numerical_resonance_models.PlanarResonanceEquations>` class then
models the dynamics of the resonant Hamiltonian,

.. math::
        {\cal H}_\mathrm{res}(\sigma_i,I_i;P , L) = H_\mathrm{Kep}(I_i ; P, L) + \bar{H}_\mathrm{int}(\sigma_i,I_i;P , L)

where :math:`P` is now a conserved quantity because :math:`{\cal
H}_\mathrm{res}` is independent of the coordinate :math:`Q`.

Units and Dynamical Variables 
*****************************

As written, :math:`{\cal H}_\mathrm{res}(\sigma_i,I_i;P , L)` represents a two
degree-of-freedom system that depends on two parameters :math:`P \mathrm{ and }
L` (in addition to the planets' and star's masses).  The
:class:`PlanarResonanceEquations
<celmech.numerical_resonance_models.PlanarResonanceEquations>` class reduces
the number of free parameters by selecting appropriate units for action
variables and time.  In particular, the units are chosen by first defining the
reference semi-major axes, :math:`a_{i,0}`, so that

.. math::
        L = \tilde{m}_1 \sqrt{G\tilde{M}_1a_{1,0}} + \tilde{m}_2 \sqrt{G\tilde{M}_2a_{2,0}}\\
        a_{1,0}/a_{2,0} = \left(\frac{j-k}{j}\right)^{2/3}\left(\frac{\tilde{M}_1}{\tilde{M}_2}\right)^{1/3}

Next, the action variables are rescaled by 

.. math::
        I_i \rightarrow \frac{1}{(\tilde{m}_1 + \tilde{m}_2)\sqrt{GM_*a_{2,0}}}I_i
        
and time is measured in units such that
:math:`\sqrt{\frac{GM_*}{a_{2,0}^{3}}}=1` (i.e., the orbital period at
:math:`a=a_{2,0}` is equal to :math:`2\pi`).  Finally, rather than specifying
the conserved quantity :math:`P`, the equations of motion used by
:class:`PlanarResonanceEquations
<celmech.numerical_resonance_models.PlanarResonanceEquations>` are formulated
in terms of 

.. math::
        {\cal D} = \frac{\Lambda_{2,0}-\Lambda_{1,0} - \Lambda_{2} + \Lambda_{1}}{s+1/2} + \Gamma_1 + \Gamma_2

where :math:`\Lambda_{i,0} = \tilde{m}_i \sqrt{G\tilde{M}_ia_{i,0}}`.  In terms
of orbital elements, 

.. math::
        {\cal D}\approx
        \beta_{1}\sqrt{\alpha_0}\left(1-\sqrt{1-e_1^2}\right)
        +
        \beta_{2}\left(1-\sqrt{1-e_2^2}\right)
        -
        \frac{k\beta_2\beta_1 \sqrt{\alpha_0} \Delta }{3 \left(\beta_1\sqrt{\alpha_0}  (s+1)+\beta_2 s\right)}

where :math:`\Delta = \frac{j-k}{j}\frac{P_2}{P_1} - 1`, :math:`\beta_i =
\frac{\tilde{m}_i}{\tilde{m}_1+\tilde{m}_2}\sqrt{\frac{\tilde{M}_i}{M_*}}` and
:math:`\alpha_0 = \frac{a_{1,0}}{a_{2,0}}`.

Finally, :class:`PlanarResonanceEquations
<celmech.numerical_resonance_models.PlanarResonanceEquations>` uses canonical
coordinate :math:`y_i = \sqrt{2I_i}\sin\sigma_i` and conjugate momenta
:math:`x_i = \sqrt{2I_i}\cos\sigma_i` instead of :math:`(\sigma_i,I_i)` in
order to formulate the equations of motion.

Various methods of the :class:`PlanarResonanceEquations
<celmech.numerical_resonance_models.PlanarResonanceEquations>` class take
dynamical variables as a vector in the standardized form:

.. math::
        \pmb{z} = (y_1,y_2,x_1,x_2,{\cal D})

when computing equations of motion and other quantities.



Dissipative Forces
******************

 The :class:`PlanarResonanceEquations
 <celmech.numerical_resonance_models.PlanarResonanceEquations>` class includes
 capabilities to model dissipative forces inducing migration and eccentricity
 damping in addition to the conservative dynamics of resonant interactions.
 Specifically, migration forces of the form

.. math::
        \begin{align}
        \frac{d\ln a_i}{dt} &= -\frac{1}{\tau_{m,i}} - 2p\frac{e_i^2}{\tau_{e,i}}\\
        \frac{d\ln e_i}{dt} &= -\frac{1}{\tau_{e,i}}
        \end{align}

These forces are specified by setting the attributes :attr:`K1
<celmech.numerical_resonance_models.PlanarResonanceEquations.K1>`, :attr:`K2
<celmech.numerical_resonance_models.PlanarResonanceEquations.K2>`, :attr:`p
<celmech.numerical_resonance_models.PlanarResonanceEquations.p>`, and
:attr:`tau_alpha
<celmech.numerical_resonance_models.PlanarResonanceEquations.tau_alpha>`.

Spatial Resonance Equations
---------------------------

The :class:`~celmech.numerical_resonance_models.SpatialResonanceEquations`
class provides equations of motion governing a mean-motion resonance between a
planet pair on mutually inclined orbits.  To study the dynamics of a
:math:`j:j-k` MMR, we will perform a canonical transformation from the usual
:ref:`Poincare canonical variables<poincare>` to new canonical variables as in
the planar case above.  However, before proceeding with this transformation, we
follow `Malige (2002)
<https://ui.adsabs.harvard.edu/abs/2002CeMDA..84..283M/abstract>`_ and use the
invariance of the system's angular momentum vector direction to reduce the
number of degrees of freedom by one.  This reduction is accomplished by first
adopting a coordinate system in which the system's angular momentum vector lies
along the :math:`z`-axis.  Next, complex canonical variables
:math:`y_i=\sqrt{Q_i}e^{-\mathrm{i} q_i}` are introduced and the reduced
Hamiltonian is written by replacing the complex canonical variables :math:`y_i`
with the expressions

.. math::
        \begin{eqnarray}
            y_1 &=& \mathrm{i} \frac{y}{\sqrt{2}}
            \sqrt{1 +
                \frac{\Lambda_2 - \Gamma_2 - \Lambda_1 + \Gamma_1}
                {\Lambda_1 + \Lambda_2 - \Gamma_2 - \Gamma_1 - y\bar{y}}
            }
            \\
            y_2 &=& -\mathrm{i} \frac{y}{\sqrt{2}}
            \sqrt{1 +
                \frac{\Lambda_2 - \Gamma_2 - \Lambda_1 + \Gamma_1}
                {\Lambda_1 + \Lambda_2 - \Gamma_2 - \Gamma_1 - y\bar{y}}
            }~,
        \end{eqnarray}

which introduce the new complex canonical variable :math:`y`. Finally, the new
canonical variable :math:`y` can be written as
:math:`y=\sqrt{Q}e^{-\mathrm{i}q}`, introducing the canonical
coordinate-momentum pair :math:`(q,Q)`. In terms of orbital elements, the new
variables are

.. math::
        Q=2\mu_1\sqrt{M_1a_1(1-e_1^2)}\sin^2(I_1/2) +2\mu_2\sqrt{M_2a_2(1-e_2^2)}\sin^2(I_2/2)

and :math:`q=-\frac{1}{2}(\Omega_1+\Omega_2)`.

Now we introduce the new canonical angle variables

.. math::
        \begin{align}
            \begin{pmatrix}\sigma_1\\ \sigma_2 \\ \phi \\ \psi \\ l  \end{pmatrix}
            =
        \begin{pmatrix} 
        -s & 1+s & 1 & 0 & 0 \\
        -s & 1+s & 0 & 1 & 0 \\
        -s & 1+s & 0 & 0 & 1 \\
        -1/k & 1/k & 0 & 0 & 0 \\
        -s & 1+s & 0 & 0 & 0 
        \end{pmatrix}
        \cdot
        \begin{pmatrix}\lambda_1 \\ \lambda_2 \\ \gamma_1 \\ \gamma_2 \\ q  \end{pmatrix}
        \label{eq:equations_of_motion:new_angles}
        \end{align}

where :math:`s = (j-k)/{k}`, along with their conjugate action variables

  .. math::
        \begin{align}
            \begin{pmatrix}I_1 \\ I_2 \\ \Phi \\ \Psi \\ L  \end{pmatrix}
            =
        \begin{pmatrix} 
        0 & 0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 0 & 1 \\
        -j & k-j & 0 & 0 & 0 \\
        1 & 1 & -1 & -1 & -1 
        \end{pmatrix}
        \cdot
        \begin{pmatrix}\Lambda_1 \\ \Lambda_2 \\ \Gamma_1 \\ \Gamma_2 \\ Q  \end{pmatrix}~.
        \label{eq:equations_of_motion:new_actions}
        \end{align}
        
After this canonical transformation, the Hamiltonian is independent of the
angle variables :math:`l` so the quantity

.. math::
        \begin{eqnarray}
            L &=& \Lambda_1+\Lambda_2-Q-\Gamma_1-\Gamma_2
        \end{eqnarray}

is conserved, reflecting the conservation of the system's total angular
momentum.  The phase space is further reduced by numerically averaging over the
fast angle :math:`\psi`, similar to the
:class:`PlanarResonanceEquations<celmech.numerical_resonance_models.PlanarResonanceEquations>`
class.  We also eliminate the explicit dependence of our equations of motion on
the value of the total angular momentum by introducing a simultaneous
re-scaling of the Hamiltonian and the action variables that preserves the form
of Hamilton's equations. In order to do so, let us first define the reference
semi-major axes :math:`a_{1,0}` and :math:`a_{2,0}` such that

.. math::
    \begin{equation*}
        L=\mu_1\sqrt{GM_1a_{1,0}} + \mu_2\sqrt{GM_2a_{2,0}}
    \end{equation*}

and

.. math::
    \begin{equation*}
        a_{1,0}=\left(\frac{M_1}{M_2}\right)^{1/3}\left(\frac{j-k}{j}\right)^{2/3}a_{2,0}~.
    \end{equation*}

In other words,  :math:`a_{1,0}` is defined as the nominal location of an exact
resonance with a planet situated at :math:`a_{2,0}`. We then re-scale the
Hamiltonian and the action variable, dividing each by a factor of :math:`(\mu_1
+ \mu_2)\sqrt{GM_*a_{2,0}}` so that the new Hamiltonian and canonical action
variables are given by

.. math::
        \begin{eqnarray}
            \{\mathcal{H},{I}_1,{I}_2,{\Phi},{\Psi}\}
           \rightarrow  
                \frac{1}{(\mu_1+\mu_2)\sqrt{GM_*a_{2,0}}}
                    \times
                \{\mathcal{H},I_1,I_2,\Phi,\Psi\}~.
        \end{eqnarray}

and the re-scaled angular momentum, :math:`{L}= \beta_2 +
\beta_1\sqrt{a_{1,0}/a_{2,0}}`, is a constant.  In terms of the newly re-scaled
variables, the averaged Hamiltonian is then

.. math::
    \mathcal{H}(\sigma_1,\sigma_2,\phi,{I}_1,{I}_2,{\Phi};{\Psi}) =
    H_\mathrm{kep}({I}_1,{I}_2,{\Phi};{\Psi}) + 
    \epsilon \bar{H}_\mathrm{int}(\sigma_1,\sigma_2,\phi,{I}_1,{I}_2,{\Phi};{\Psi})

where 

.. math::
    \begin{eqnarray}
    H_\mathrm{kep} &=& -n_{2,0}\sum_{i=1}^2\frac{\beta_i^3(1+m_i/M_*)^{1/2}}{2{\Lambda}_i^2}\\
     \bar{H}_\mathrm{int}&=&\frac{n_{2,0}}{2\pi}\int_{-\pi}^{\pi}\left(\frac{a_{2,0}}{GM_*}\pmb{v}_1\cdot\pmb{v}_2-\frac{a_{2,0}}{|\pmb{r}_2-\pmb{r}_1|}  \right)d\psi
    \end{eqnarray}

with
:math:`\beta_i=\frac{\mu_i}{\mu_1+\mu_2}\sqrt{1+m_i/M_*}`, 
:math:`n_{2,0} = \sqrt{GM_*a_{2,0}^{-3}}`,
:math:`\epsilon = \frac{m_1m_2}{M_*(\mu_1+\mu_2)}`, and

.. math::
    \begin{align*}
        {\Lambda}_1 &= -s{L}-{\Psi}/k - s({I}_1 + {I}_2 + {\Phi})= \beta_1\sqrt{a_{1}/a_{2,0}}\\
        {\Lambda}_2 &= (s+1){L} + {\Psi}/k + (s+1)({I}_1 + {I}_2 + {\Phi})= \beta_2\sqrt{a_{2}/a_{2,0}}~.
    \end{align*}

In order to avoid coordinate singularities that occur when :math:`I_i=0` or
:math:`\Phi=0`, the
:class:`~celmech.numerical_resonance_models.SpatialResonanceEquations` class
formulates the Hamiltonian in terms of the canonical variables

.. math::
    \begin{eqnarray}
    (y_i,x_i) &=& (\sqrt{2I_i}\sin\sigma_i,\sqrt{2I_i}\cos\sigma_i)\nonumber\\
    (y_\mathrm{inc},x_\mathrm{inc}) &=& (\sqrt{2\Phi}\sin\phi,\sqrt{2\Phi}\cos\phi)~.
    \end{eqnarray}

Because the averaged Hamiltonian is independent of the angle variable
:math:`\psi`, the action variable :math:`\Psi` is a constant of motion.  Rather
than parameterizing the Hamiltonian in terms of :math:`\Psi`, we introduce a
variable :math:`{\cal D}` that has a more straightforward interpretation as the
"average" angular momentum deficit of the resonant planet pair. In order to
define :math:`{\cal D}`, let us first introduce :math:`\rho_\mathrm{res}` as
the value of :math:`{\Lambda}_1/{\Lambda}_2` when the planet pair's period
ratio is equal to the nominal resonant period ratio. Explicitly,
:math:`\rho_\mathrm{res} =
\frac{\beta_1}{\beta_2}\left(\frac{j-k}{j}\right)^{1/3}\left(\frac{M_1}{M_2}\right)^{1/6}`.
Then, we define :math:`{\cal D}`, along with an additional constant
:math:`\Lambda_{2,\mathrm{res}}`, such that

.. math::
        \begin{equation}
            \begin{pmatrix}
             -\Psi/k\\
                {L}
            \end{pmatrix}
            = 
            \begin{pmatrix}
            s + (1+s) \rho_\mathrm{res} & 0 \\
            (1+\rho_\mathrm{res}) & -1
        \end{pmatrix}
        \cdot
        \begin{pmatrix}
            \Lambda_{2,\text{res}} \\
            {\cal D}
        \end{pmatrix}~.
        \end{equation}

Inverting this equation and solving for :math:`\mathcal{D}` yields

.. math::
    \begin{equation}
         {\cal D} = \frac{1}{2}\left(x_1^2+y_1^2+x_2^2+y_2^2+x_\mathrm{inc}^2+y_\mathrm{inc}^2\right) + (\Lambda_{2,\mathrm{res}}-{\Lambda}_2) + (\rho_\mathrm{res}\Lambda_{2,\mathrm{res}}-{\Lambda}_1) 
    \end{equation}

We express :math:`\mathcal{D}` in terms of the planet pair's period ratio by using

.. math:: \Delta \equiv \frac{j-k}{j}\frac{P_2}{P_1} - 1 \approx
   3\left(\frac{\delta\hat{\Lambda}_{2}}{\Lambda_{2,\mathrm{res}}}-\frac{\delta\hat{\Lambda}_{1}}{\rho_\mathrm{res}\Lambda_{2,\mathrm{res}}}\right)=3\frac{\delta\hat{\Lambda}_{2}}{\Lambda_{2,\mathrm{res}}}\left(1+\frac{s}{(1+s)\rho_\mathrm{res}}\right) 
        
allowing us to write 

.. math::
    \begin{eqnarray}
    {\cal D} = \frac{1}{2}\left(x_1^2+y_1^2+x_2^2+y_2^2+x_\mathrm{inc}^2+y_\mathrm{inc}^2\right) - \frac{1}{3}
    \frac{\rho_\mathrm{res}\Lambda_{2,\mathrm{res}}}{s + \rho_\mathrm{res}(1+s)} \Delta
    \end{eqnarray}

In other words, :math:`\mathcal{D}` sets the value of the pair's angular momentum deficit when the planets' period ratio is at the exact resonant value (:math:`\Delta=0`).

API
---

.. autoclass:: celmech.numerical_resonance_models.PlanarResonanceEquations
        :members:

.. autoclass:: celmech.numerical_resonance_models.SpatialResonanceEquations
        :members:
