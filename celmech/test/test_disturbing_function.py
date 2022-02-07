import rebound
import unittest
import math
import numpy as np
from celmech.disturbing_function import laplace_b, df_coefficient_C,df_coefficient_Ctilde, get_fg_coefficients, evaluate_df_coefficient_dict
from random import random, seed

class TestDisturbingFunction(unittest.TestCase):

    def setUp(self):
        self.alpha = 0.5
        self.LaskarRobutel=dict()
        self.LaskarRobutel['C1'] = {(0,(1/2,0,0)):1/2}
        self.LaskarRobutel['C2'] = {
            (1,(3/2,0,0)):+3/8,
            (0,(3/2,1,0)):-1/4,
            (2,(3/2,1,0)):-1/4
        }
        self.LaskarRobutel['C3'] = {
            (1,(3/2,1,0)):1/4,
        }
        self.LaskarRobutel['C4'] = {
            (1,(5/2,0,0)):-15/4,
            (3,(5/2,0,0)):-15/4,
            (0,(5/2,1,0)):3/2,
            (2,(5/2,1,0)):27/8,
            (4,(5/2,1,0)):3/2,
        }
        self.LaskarRobutel['C7'] = {
            (1,(5/2,0,0)):15/8,
            (3,(5/2,0,0)):15/8,
            (0,(5/2,1,0)):-3/4,
            (2,(5/2,1,0)):-9/8,
            (4,(5/2,1,0)):-3/4,
        }
        self.LaskarRobutel['C8'] = {
            (2,(5/2,1,0)):9/16
        }
        self.LaskarRobutel['C9'] = {
            (2,(5/2,0,0)):-15/32,
            (1,(5/2,1,0)):3/16,
            (3,(5/2,1,0)):9/16,
        }
        self.LaskarRobutel['C10'] = {
            (2,(5/2,0,0)):+45/128,
            (1,(5/2,1,0)):-9/64,
            (3,(5/2,1,0)):-9/64
        }
        self.LaskarRobutel['C11'] = {
            (2,(5/2,0,0)):3/8,
            (1,(5/2,1,0)):-3/4,
            (3,(5/2,1,0)):-3/4,
        }
        self.LaskarRobutel['C12'] = {
            (2,(5/2,0,0)):-15/4,
            (1,(5/2,1,0)):3/4,
            (3,(5/2,1,0)):3/4,
        }
        self.LaskarRobutel['C13'] = {
            (2,(5/2,0,0)):+9/128,
            (1,(5/2,1,0)):-3/64,
            (3,(5/2,1,0)):+3/64
        }
        self.LaskarRobutel['C14'] = {
            (2,(5/2,0,0)):+21/8,
            (1,(5/2,1,0)):-3/4,
            (3,(5/2,1,0)):-3/4
        }
        self.LaskarRobutel['C15'] = {
            (2,(5/2,0,0)):3/8,
            (1,(5/2,1,0)):3/8,
            (3,(5/2,1,0)):3/8
        }
        self.LaskarRobutel['C17'] = {
            (2,(5/2,0,0)):-15/32,
            (1,(5/2,1,0)):9/16,
            (3,(5/2,1,0)):3/16,
        }
        self.LaskarRobutel['C18'] = {
            (2,(5/2,0,0)):9/8,
        }
    def tearDown(self):
        self.sim = None
    def compare_objects(self, obj1, obj2, delta=1.e-15):
        self.assertEqual(type(obj1), type(obj2))
        for attr in [attr for attr in dir(obj1) if not attr.startswith('_')]:
            self.assertAlmostEqual(getattr(obj1, attr), getattr(obj2, attr), delta=delta)

    def compare_DF_coeffs(self,df1,df2,delta=1.e-12):
        val1 = evaluate_df_coefficient_dict(df1,self.alpha)
        val2 = evaluate_df_coefficient_dict(df2,self.alpha)
        self.assertAlmostEqual(val1,val2,delta=delta)

    def test_df_coefficient_C(self):
        # cos[0]
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                self.LaskarRobutel['C1'],
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # e1*e1*cos[w1-w2]
        j1,j2,j3,j4,j5,j6 = 0,0,1,-1,0,0
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C2'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # e1^2
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        z1,z2,z3,z4  = 0,0,1,0
        self.compare_DF_coeffs(
                {key:val/2 for key,val in self.LaskarRobutel['C3'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # e1*e*inc1*inc*cos(w-w1+Omega-Omega1)
        j1,j2,j3,j4,j5,j6 = 0,0,1,-1,1,-1
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C4'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # e1*e*inc^2*cos(w-w1)
        j1,j2,j3,j4,j5,j6 = 0,0,1,-1,0,0
        z1,z2,z3,z4  = 1,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C7'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # e1*e*inc1^2*cos(w-w1)
        j1,j2,j3,j4,j5,j6 = 0,0,1,-1,0,0
        z1,z2,z3,z4  = 0,1,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C7'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )

        # C8
        j1,j2,j3,j4,j5,j6 = 0,0,+1,+1,-1,-1
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:4*val for key,val in self.LaskarRobutel['C8'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        j1,j2,j3,j4,j5,j6 = 0,0,-1,+1,+1,-1
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:-4*val for key,val in self.LaskarRobutel['C8'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        j1,j2,j3,j4,j5,j6 = 0,0,-1,-1,2,0
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:-2*val for key,val in self.LaskarRobutel['C8'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        j1,j2,j3,j4,j5,j6 = 0,0,-1,-1,0,2
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:-2*val for key,val in self.LaskarRobutel['C8'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # C9
        j1,j2,j3,j4,j5,j6 = 0,0,0,-2,2,0
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C9'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        j1,j2,j3,j4,j5,j6 = 0,0,0,-2,0,2
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C9'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        j1,j2,j3,j4,j5,j6 = 0,0,0,-2,1,1
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:-4*val for key,val in self.LaskarRobutel['C9'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # C11
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        z1,z2,z3,z4  = 1,0,1,0
        self.compare_DF_coeffs(
                {key:val for key,val in self.LaskarRobutel['C11'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        z1,z2,z3,z4  = 0,1,0,1
        self.compare_DF_coeffs(
                {key:val for key,val in self.LaskarRobutel['C11'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # C12
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,1,-1
        z1,z2,z3,z4  = 1,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C12'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,1,-1
        z1,z2,z3,z4  = 0,1,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C12'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )

        # e1^2e2^2 cos(2w1-2w2)
        j1,j2,j3,j4,j5,j6 = 0,0,2,-2,0,0
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C10'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # e1^4
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        z1,z2,z3,z4  = 0,0,2,0
        self.compare_DF_coeffs(
                self.LaskarRobutel['C13'],
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # inc^4
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        z1,z2,z3,z4  = 2,0,0,0
        self.compare_DF_coeffs(
                self.LaskarRobutel['C14'],
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        self.compare_DF_coeffs(
                self.LaskarRobutel['C14'],
                df_coefficient_C(j1,j2,j3,j4,j5,j6,0,2,z3,z4)
        )
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,2,-2
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C14'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )

        # C15
        # inc1 inc2 e^2 cos(Omega1-Omega2)
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,1,-1
        z1,z2,z3,z4  = 0,0,1,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C15'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,1,-1
        z1,z2,z3,z4  = 0,0,0,1
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C15'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        # C17
        j1,j2,j3,j4,j5,j6 = 0,0,-2,0,2,0
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C17'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        j1,j2,j3,j4,j5,j6 = 0,0,-2,0,0,2
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:2*val for key,val in self.LaskarRobutel['C17'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        j1,j2,j3,j4,j5,j6 = 0,0,-2,0,1,1
        z1,z2,z3,z4  = 0,0,0,0
        self.compare_DF_coeffs(
                {key:-4*val for key,val in self.LaskarRobutel['C17'].items()},
                df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
        )
        
        # e1^2e2^2 
        j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0
        for zvec, factor in zip([(0,0,1,1),(1,1,0,0),(1,0,0,1),(0,1,1,0)],[0.25,4,-1,-1]):
            z1,z2,z3,z4  = zvec
            self.compare_DF_coeffs(
                    {key:factor*val for key,val in self.LaskarRobutel['C18'].items()},
                    df_coefficient_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
            )
    def test_get_fg_coffs(self):
        f,g = get_fg_coefficients(17,3)
        self.assertAlmostEqual(f,-5.603736926452656)
        self.assertAlmostEqual(g,6.2337962206883) 

if __name__ == '__main__':
    unittest.main()

