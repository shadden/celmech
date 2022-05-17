.. _disturbing_function:

Disturbing Function
===================

Introduction
------------

The full  Hamiltonian for a system of :math:`N` planets can be written in terms
of canonical heliocentric variables as 

.. math::

        H = -\sum_{i=1}^N\frac{G^2(M_*+m_i)^2\mu_i^3}{2\Lambda_i^2} + \sum_{i=1}^N\sum_{j=1}^i \left(
        -\frac{Gm_im_j}{|\pmb{r}_i-\pmb{r}_j|} + \frac{\tilde {\bf r}_i \cdot \tilde {\bf r}_j}{m_0}
        \right)

The second term, representing the gravitational interaction potential between
the planets in the system, makes the problem non-integrable.  Pairwise
interactions between two planets, :math:`i` and :math:`j`, can be expressed in
terms of the so-called 'disturbing function', :math:`{\cal R}^{(i,j)}`, defined
so that

.. math::

        -\frac{Gm_im_j}{|\pmb{r}_i-\pmb{r}_j|} + \frac{\tilde {\bf r}_i \cdot \tilde {\bf r}_j}{m_0} = -\frac{Gm_im_j}{a_j}{\cal R}^{(i,j)}~.

The disturbing function, :math:`{\cal R}^{(i,j)}`, can be expanded in a cosine
series in the planets' angular orbital elements.  Defining
:math:`\pmb{\theta}_{i,j} =
(\lambda_j,\lambda_i,\varpi_i,\varpi_j,\Omega_i,\Omega_j)`, we can write this
cosine series as

.. math::

        {\cal R}^{(i,j)} = \sum_{\bf k}c_{\pmb{k}}(\alpha,e_i,e_j,I_i,I_j)\cos(\pmb{k}\cdot \pmb{\theta}_{i,j})

where :math:`\alpha = a_i/a_j`.  Rotation and reflection symmetries of the
planets' gravitational interactions dictate that :math:`c_{\bf k}\ne 0` only if
:math:`\sum_{l=1}^{6}k_l = 0` and :math:`k_5+k_6=2n` where :math:`n` is an
integer.

Expansion in orbital elements
*****************************

In classical disturbing function expansions, the cosine amplitudes
:math:`c_{\pmb{k}}` are further expanded as Taylor series in powers of the
planets' eccentricities :math:`e` and :math:`s = \sin(I_i/2)` as

.. math::

   c_{\bf k}(\alpha,e_i,e_j,I_i,I_j) = e_i^{|k_3|}e_j^{|k_4|}s_i^{|k_5|}s_j^{|k_6|}
                                       \sum_{\nu_1,\nu_2,\nu_3,\nu_4=0}^\infty 
                                       \tilde{C}_{\bf k}^{(\nu_1,\nu_2,\nu_3,\nu_4)}(\alpha)
                                       s_i^{2\nu_1}s_j^{2\nu_2}e_i^{2\nu_3}e_j^{2\nu_4}

``celmech`` offers the capability to compute the individual disturbing function
coefficients through the function
:func:`~celmech.disturbing_function.df_coefficient_Ctilde`.

Expansion in canonical variables
********************************

Since ``celmech`` formulates equations of motion in terms of canonical
variables rather than orbital elements, it is often more convenient to
formulate the Taylor series expansion of the disturbing function as 

.. math::

                c_{\bf k}(\alpha,e_i,e_j,I_i,I_j)\exp[i\pmb{k}\cdot\pmb{\theta}_{i,j}] = 
                \exp[i(k_1\lambda_j + k_2\lambda_i)]
                \bar{X_i}^{-k_3}
                \bar{X_j}^{-k_4}
                \bar{Y_i}^{-k_5}
                \bar{Y_j}^{-k_6}
                \\
                \times\sum_{\nu_1,\nu_2,\nu_3,\nu_4=0}^\infty 
                \hat{C}_{\bf k}^{(\nu_1,\nu_2,\nu_3,\nu_4)}(\alpha_{ij})
                |Y_i|^{2\nu_1}
                |Y_j|^{2\nu_2}
                |X_i|^{2\nu_3}
                |X_j|^{2\nu_4},

where 

.. math::
        \begin{align}
        X_i &=& \sqrt{\frac{2}{\Lambda_i}}(\kappa_i - \sqrt{-1}\eta_i)\approx e_ie^{\sqrt{-1}\varpi_i} + \mathcal{O}(e_i^3)\\
        Y_i &=& \sqrt{\frac{1}{2\Lambda_i}}(\sigma_i - \sqrt{-1}\rho_i)\approx s_ie^{\sqrt{-1}\Omega_i}+ \mathcal{O}(s_ie_i^2)
        \end{align}

and bars denote complex conjugates. Writing the expansion above, we've assumed
:math:`k_3,k_4,k_5,k_6 \le 0`. If :math:`k_3>0` then one makes the replacement
:math:`\bar{X_i}^{-k_3}\rightarrow {X_i}^{k_3}` and a similar replacements are
made if :math:`k_4,k_5,` or :math:`k_6>0`.  Note that :math:`\hat{C}_{\bf
k}^{(0,0,0,0)} = \tilde{C}_{\bf k}^{(0,0,0,0)}`. The relationship between the
coefficients :math:`\hat{C}_{\bf k}^{\pmb{\nu}}` and :math:`\tilde{C}_{\bf
k}^{\pmb{\nu}}` when :math:`\pmb{\nu}\ne0` is more complicated.

Complete expansion
******************

Often, the explicit dependence of the disturbing function on the variables
:math:`\Lambda_i` is ignored. This allows the resulting equiations of motion to
be expressed as polynomials with fixed coefficients involving the variables
:math:`\eta_i,\kappa_i,\rho_i,\sigma_i` and various trignometric functions of
the :math:`\lambda_i` variables. At this level of approximation, Hamilton's
equations for :math:`\lambda_i` become :math:`\dot{\lambda_i} =
\frac{\partial}{\partial \Lambda_i}H\approx \frac{\partial}{\partial
\Lambda_i}H_\mathrm{Kep}`. We can construct a more accurate model while
retaining the polynomial nature of the equations of motion by Taylor-expanding
the disturbing function coefficients :math:`{C}_{\bf
k}^{\pmb{\nu}}(\alpha_{i,j})` with respect to the :ref:`Poincare <poincare_intro>`
momenta :math:`\Lambda_i` about some reference values :math:`\Lambda_{i,0}`.
Defining :math:`\delta_i = (\Lambda_i-\Lambda_{i,0}) / \Lambda_{i,0}`, we write

.. math::
        \hat{C}_{\bf k}^{\pmb{\nu}}(\alpha_{ij}) = \sum_{l_1,l_2 = 0}^\infty {C}_{\bf k}^{\pmb{\nu},(l_1,l_2)}(\alpha_{ij,0})\delta_i^{l_1}\delta_j^{l_2}

The complete expansion of the interaction term between planets :math:`i` and
:math:`j` can therefore be written as

.. math::
        \begin{multline}
        -\frac{Gm_im_j}{|\mathbf{r}_i - \mathbf{r}_j|} + \frac{\tilde{\mathbf{r}}_i \cdot \tilde{\mathbf{r}}_j}{m_*}
         = \\
         -\frac{Gm_im_j}{a_{j,0}}\sum_{\mathbf{k}}\sum_{\nu_1,...,\nu_4 = 0}^\infty \sum_{l_1,l_2=0}^\infty 
      C_{\bf{k}}^{\pmb{\nu},\pmb{l}}(\alpha_{ij,0})
      |Y_i|^{|k_5|+2\nu_1}
      |Y_j|^{|k_6|+2\nu_2}
      |X_i|^{|k_3|+2\nu_3}
      |X_j|^{|k_4|+2\nu_4}
      \delta_i^{l_1}\delta_j^{l_j}
      \\
      \times \cos(k_1\lambda_j+k_2\lambda_i+k_3\varpi_i+k_4\varpi_j+k_5\Omega_i+k_6\Omega_j)~.
      \end{multline}

Individual coefficients, :math:`C_{\bf{k}}^{\pmb{\nu},\pmb{l}}`, can be
obtained as dictionaries with the function
:func:`~celmech.disturbing_function.df_coefficient_C`. This is the basic
function that the various methods for adding terms to a
:class:`~celmech.poincare.PoincareHamiltonian` are built upon.

Calculating Coefficients
------------------------

The functions :func:`df_coefficient_Ctilde()
<celmech.disturbing_function.df_coefficient_Ctilde>` and
:func:`df_coefficient_C()<celmech.disturbing_function.df_coefficient_C>`
calculate the disturbing function coefficients :math:`\tilde{C}_{\bf
k}^{\pmb{\nu}}` and :math:`{C}_{\bf k}^{\pmb{\nu},(l_1,l_2)}`, respectively.
These functions return results as dictionaries representing sums of Laplace
coefficients,

.. math::
        b_{s}^{(j)}(\alpha) = \frac{1}{\pi}\int_{0}^\pi \frac{\cos(j\theta)}{(1 + \alpha^2 - 2\alpha\cos\theta)^{s}},

and their derivatives with respect to :math:`\alpha`.  Specifically, the
dictionary keys represent Laplace coefficients and values represent their
coefficients. For example, the disturbing function coefficient 
       
.. math:: 
        \tilde{C}_{(5,-2,0,-1,-2,0)}^{(0,0,0,0)}(\alpha) = \frac{10}{4}\alpha
   b_{3/2}^{(3)} + \frac{1}{4}\alpha^2\frac{d}{d\alpha}b_{3/2}^{(3)}

is obtained with :func:`~celmech.disturbing_function.df_coefficient_Ctilde`
using 

.. code:: python3

   k = (5,-2,0,-1,-2,0)
   nu = (0,0,0,0)
   Ccoeff = df_coefficient_Ctilde(*k,*nu)
   print(Ccoeff)

which dispays::
        
        {(1, (1.5, 3, 0)): 2.5, (2, (1.5, 3, 1)): 0.25, ('indirect', 1): 0} 

The value of the disturbing function coefficient as a function of
:math:`\alpha` is given by the sum of dictionary key-value pairs
:math:`\{(p_i,(s_i,j_i,n_i)): A_i\}` and
:math:`\{(\mathrm{'indirect'},p_\mathrm{ind}):A_\mathrm{ind}\}` according to 

.. math::
        \sum_{i}A_i \alpha^p\frac{d}{d\alpha}b^{(j)}_{s}(\alpha) + A_\mathrm{ind}\alpha^{-p_\mathrm{ind}/2}~.

These dictionaries can be evaluated at a specific semi-major axis ratio,
:math:`\alpha`, using the function
:func:`~celmech.disturbing_function.evaluate_df_coefficient_dict`.  Note that
individual Laplace coefficient values can also be evaluated using the
:func:`~celmech.disturbing_function.laplace_b` function.

Generating Arguments
--------------------

In many applications, a collection of disturbing function terms with related
arguments are desired.  For example, a user might want all of the arguments
corresponding to a particular mean motion resonance or all of the secular terms
up to a given order in some planet pair's eccentricities.  The
`disturbing_function` module provides methods to generate lists of disturbing
function arguments (i.e., :math:`(\pmb{k},\pmb{\nu})` combinations) related to
one another.

The function :func:`~celmech.disturbing_function.df_arguments_dictionary`
supplies a comprehensive list of all possible disturbing function cosine
arguments appearing up to a given order, :math:`N_\mathrm{max}`, in planets'
eccentricities and inclinations.  The functions
:func:`~celmech.disturbing_function.list_secular_terms` and
:func:`~celmech.disturbing_function.list_resonance_terms` can provide lists of
:math:`(\pmb{k},\pmb{\nu})` pairs for secular or resonant disturbing function
terms in a user-specified range of orders.

API
---
.. automodule:: celmech.disturbing_function
    :members: df_coefficient_Ctilde, df_coefficient_C, evaluate_df_coefficient_dict, laplace_b,df_arguments_dictionary, list_resonance_terms, list_secular_terms
