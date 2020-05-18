import rebound
import unittest
import math
import numpy as np
from celmech.disturbing_function import laplace_b, DFCoeff_C,DFCoeff_Cbar, get_fg_coeffs, eval_DFCoeff_dict
from random import random, seed

class TestDisturbingFunction(unittest.TestCase):

    def setUp(self):
        self.alpha = 0.5
        self.LaskarRobutel_C2 = {
            (1,(3/2,0,0)):3/8,
            (0,(3/2,1,0)):-1/4,
            (2,(3/2,1,0)):-1/4
        }
    def tearDown(self):
        self.sim = None
    def compare_objects(self, obj1, obj2, delta=1.e-15):
        self.assertEqual(type(obj1), type(obj2))
        for attr in [attr for attr in dir(obj1) if not attr.startswith('_')]:
            self.assertAlmostEqual(getattr(obj1, attr), getattr(obj2, attr), delta=delta)
    def compare_DF_coeffs(self,df1,df2,delta=1.e-12):
        val1 = eval_DFCoeff_dict(df1,self.alpha)
        val2 = eval_DFCoeff_dict(df2,self.alpha)
        self.assertAlmostEqual(val1,val2,delta=delta)
    def test_DFCoeff_C(self):
        # cos[0]
        LaskarRobutel_C1 = {(0,(1/2,0,0)):1/2}
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                LaskarRobutel_C1,
                DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # e1*e1*cos[w1-w2]
        LaskarRobutel_C2 = {
            (1,(3/2,0,0)):+3/8,
            (0,(3/2,1,0)):-1/4,
            (2,(3/2,1,0)):-1/4
        }
        j1,j2,j3,j4,j5,j6 = 0,0,1,-1,0,0
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in LaskarRobutel_C2.items()},
                DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # e1^2
        LaskarRobutel_C3 = {
            (1,(3/2,1,0)):1/4,
        }
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        z1,z2,z3,z4  = 0,0,1,0
        self.compare_DF_coeffs(
                {key:val/2 for key,val in LaskarRobutel_C3.items()},
                DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        self.compare_DF_coeffs(
                DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4),
                DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,0,1)
        )
        # e1^2e2^2 cos(2w1-2w2)
        LaskarRobutel_C10 = {
            (2,(5/2,0,0)):+45/128,
            (1,(5/2,1,0)):-9/64,
            (3,(5/2,1,0)):-9/64
        }
        j1,j2,j3,j4,j5,j6 = 0,0,2,-2,0,0
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in LaskarRobutel_C10.items()},
                DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # e1^4
        LaskarRobutel_C13 = {
            (2,(5/2,0,0)):+9/128,
            (1,(5/2,1,0)):-3/64,
            (3,(5/2,1,0)):+3/64
        }
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        z1,z2,z3,z4  = 0,0,2,0
        self.compare_DF_coeffs(
                LaskarRobutel_C13,
                DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # inc^4
        LaskarRobutel_C14 = {
            (2,(5/2,0,0)):+21/8,
            (1,(5/2,1,0)):-3/4,
            (3,(5/2,1,0)):-3/4
        }
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        z1,z2,z3,z4  = 2,0,0,0
        self.compare_DF_coeffs(
                LaskarRobutel_C14,
                DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        self.compare_DF_coeffs(
                LaskarRobutel_C14,
                DFCoeff_C(j1,j2,j3,j4,j5,j6,0,2,z3,z4)
        )

        # inc1 inc2 e^2 cos(Omega1-Omega2)
        LaskarRobutel_C15 = {
            (2,(5/2,0,0)):3/8,
            (1,(5/2,1,0)):3/8,
            (3,(5/2,1,0)):3/8
        }
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,1,-1
        z1,z2,z3,z4  = 0,0,1,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in LaskarRobutel_C15.items()},
                DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # e1^2e2^2 
        LaskarRobutel_C18 = {
            (2,(5/2,0,0)):9/8,
        }
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        for zvec, factor in zip([(0,0,1,1),(1,1,0,0),(1,0,0,1),(0,1,1,0)],[0.25,4,-1,-1]):
            z1,z2,z3,z4  = zvec
            self.compare_DF_coeffs(
                    {key:factor*val for key,val in LaskarRobutel_C18.items()},
                    DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
            )

    def test_get_fg_coffs(self):
        f,g = get_fg_coeffs(17,3)
        self.assertAlmostEqual(f,-5.603736926452656)
        self.assertAlmostEqual(g,6.2337962206883) 


    
if __name__ == '__main__':
    unittest.main()

