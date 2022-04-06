.. _disturbing_function:

Disturbing Function
===================

Introduction
------------

The full  Hamiltonian for a system of :math:`N` planets can be written in terms of canonical heliocentric variables as 

.. math::

        H = -\sum_{i=1}^N\frac{G^2(M_*+m_i)^2\mu_i^3}{2\Lambda_i^2} + \sum_{i=1}^N\sum_{j=1}^i \left(
        -\frac{Gm_im_j}{|\pmb{r}_i-\pmb{r}_j|} + \frac{\tilde {\bf r}_i \cdot \tilde {\bf r}_j}{m_0}
        \right)

The second term, representing the gravitational interaction potential between the planets in the system, makes the problem non-integrable.
Pairwise interactions between two planets, :math:`i` and :math:`j`, can be expressed in terms of 
the so-called 'disturbing function', :math:`{\cal R}^{(i,j)}`, defined so that

.. math::

        -\frac{Gm_im_j}{|\pmb{r}_i-\pmb{r}_j|} + \frac{\tilde {\bf r}_i \cdot \tilde {\bf r}_j}{m_0} = -\frac{Gm_im_j}{a_j}{\cal R}^{(i,j)}~.

The disturbing function, :math:`{\cal R}^{(i,j)}`, can be expanded in a cosine series in the planets' angular orbital elements. 
Defining :math:`\pmb{\theta}_{i,j} = (\lambda_j,\lambda_i,\varpi_i,\varpi_j,\Omega_i,\Omega_j)`, we can write this cosine series as

.. math::

        {\cal R}^{(i,j)} = \sum_{\bf k}c_{\pmb{k}}(\alpha,e_i,e_j,I_i,I_j)\cos(\pmb{k}\cdot \pmb{\theta}_{i,j})

where :math:`\alpha = a_i/a_j`. 
Rotation and reflection symmetries of the planets' gravitational interactions dictate that :math:`c_{\bf k}\ne 0` only if :math:`\sum_{l=1}^{6}k_l = 0` and :math:`k_5+k_6=2n` where :math:`n` is an integer.

In classical disturbing function expansions, the cosine amplitudes :math:`c_{\pmb{k}}` are further expanded as Taylor series in powers of the planets' eccentricities :math:`e` and :math:`s = \sin(I_i/2)` as

.. math::

   c_{\bf k}(\alpha,e_i,e_j,I_i,I_j) = e_i^{|k_3|}e_j^{|k_4|}s_i^{|k_5|}s_j^{|k_6|}
                                       \sum_{z_1,z_2,z_3,z_4=0}^\infty 
                                       \tilde{C}_{\bf k}^{(z_1,z_2,z_3,z_4)}(\alpha)
                                       s_i^{2z_1}s_j^{2z_2}e_i^{2z_3}e_j^{2z_4}

``celmech`` offers the capability to compute the individual disturbing function coefficients through the function :func:`df_coefficient_Ctilde() <celmech.disturbing_function.df_coefficient_Ctilde>`.

Since ``celmech`` formulates equations of motion in terms of canonical variables rather than orbital elements,
it is often more convenient to formulate the Taylor series expansion of the disturbing function as 

.. math::

                c_{\bf k}(\alpha,e_i,e_j,I_i,I_j)\exp[i\pmb{k}\cdot\pmb{\theta}_{i,j}] = 
                \exp[i(k_1\lambda_j + k_2\lambda_i)]
                \bar{X_i}^{-k_3}
                \bar{X_j}^{-k_4}
                \bar{Y_i}^{-k_5}
                \bar{Y_j}^{-k_6}
                \\
                \times\sum_{z_1,z_2,z_3,z_4=0}^\infty 
                {C}_{\bf k}^{(z_1,z_2,z_3,z_4)}(\alpha)
                |Y_i|^{2z_1}
                |Y_j|^{2z_2}
                |X_i|^{2z_3}
                |X_j|^{2z_4}

, where :math:`|X_i| = \sqrt{\frac{2}{\Lambda_i}}x_i` and :math:`|Y_i| = \sqrt{\frac{1}{2\Lambda_i}}y_i` and assuming :math:`k_3,k_4,k_5,k_6 \le 0` and bars denote complex conjugates.
If :math:`k_3>0` then one makes the replacement :math:`\bar{X_i}^{-k_3}\rightarrow {X_i}^{k_3}` and  a similar replacement is made if  :math:`k_4,k_5,` or :math:`k_6>0`.
Note that :math:`{C}_{\bf k}^{(0,0,0,0)} = \tilde{C}_{\bf k}^{(0,0,0,0)}`.

When construction canonical equations of motion, it is useful to 
Taylor-expand the coefficents :math:`{C}_{\bf k}^{\pmb{\nu}}(\alpha_{i,j})` 
with respect to :ref:`Poincare <poincare_intro>` momenta :math:`\Lambda_i` about some reference values :math:`\Lambda_{i,0}`. 
Defining :math:`\delta_i = (\Lambda_i-\Lambda_{i,0}) / \Lambda_{i,0}`, we write 

.. math::
        {C}_{\bf k}^{\pmb{\nu}}(\alpha_{i,j}) = \sum_{l_1,l_2 = 0}^\infty {C}_{\bf k}^{\pmb{\nu},(l_1,l_2)}(\alpha_{i,j,0})\delta_i^{l_1}\delta_j^{l_2}

Calculating Coefficients
------------------------

The functions :func:`df_coefficient_Ctilde() <celmech.disturbing_function.df_coefficient_Ctilde>`
and :func:`df_coefficient_C()<celmech.disturbing_function.df_coefficient_C>` can be used to calculate the 
disturbing function coefficients :math:`\tilde{C}_{\bf k}^{\pmb{\nu},(l_1,l_2)}` and :math:`{C}_{\bf k}^{\pmb{\nu},(l_1,l_2)}`, respectively.
These functions return results as dictionaries representing sums of Laplace coefficients,

.. math::
        b_{s}^{(j)}(\alpha) = \frac{1}{\pi}\int_{0}^\pi \frac{\cos(j\theta)}{(1 + \alpha^2 - 2\alpha\cos\theta)^{s}},

and their derivatives with respect to :math:`\alpha`.
Specifically, the dictionary keys represent Laplace coefficients and values represent their coefficients. 
These dictionaries can be evaluated at a specific semi-major axis ratio, :math:`\alpha`, using the function 
:func:`evaluate_df_coefficient_dict()<celmech.disturbing_function.evaluate_df_coefficient_dict>`.
Note that individual Laplace coefficient values can also be evaluated using 
the :func:`laplace_b<celmech.disturbing_function.laplace_b>` function.

Generating Arguments
--------------------

In many applications, a collection of disturbing function terms with related arguments are desired.
For example, a user might want all of the arguments corresponding to a particular mean motion resonance
or all of the secular terms up to a given order in some planet pair's eccentricities.
The `disturbing_function` module provides methods to 
generate lists of disturbing function arguments (i.e., :math:`(\pmb{k},\pmb{\nu})` combinations) 
related to one another.

The function :func:`df_arguments_dictionary<celmech.disturbing_function.df_arguments_dictionary>` 
supplies a comprehensive list of all possible disturbing function cosine arguments appearing up
to a given order, :math:`N_\mathrm{max}`, in planets' eccentricities and inclinations.
The functions :func:`secular_terms_list<celmech.disturbing_function.list_secular_terms>` and
:func:`list_resonance_terms<celmech.disturbing_function.list_resonance_terms>` can provide lists
of :math:`(\pmb{k},\pmb{\nu})` pairs for secular or resonant disturbing function terms in a
user-specified range of orders.

API
---
.. automodule:: celmech.disturbing_function
    :members: df_coefficient_Ctilde, df_coefficient_C, evaluate_df_coefficient_dict, laplace_b,df_arguments_dictionary, list_resonance_terms, list_secular_terms
