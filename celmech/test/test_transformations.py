import unittest
import numpy as np
from celmech.transformations import masses_to_jacobi, masses_from_jacobi, ActionAngleToXY, XYToActionAngle

class TestTransformations(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_jacobi_masses(self):
        m = [1, 1.e-3, 1.e-3]
        mjac, Mjac = masses_to_jacobi(m)
        self.assertAlmostEqual(mjac[0], 1.002)
        self.assertAlmostEqual(mjac[1], 1.e-3/1.001)
        self.assertAlmostEqual(mjac[2], 1.e-3*1.001/1.002)
        self.assertAlmostEqual(Mjac[1], 1.001)
        self.assertAlmostEqual(Mjac[2], 1.002/1.001)

    def test_jacobi_masses_and_back(self):
        m = [2, 1.e-3, 1.e-3]
        mjac, Mjac = masses_to_jacobi(m)
        m2 = masses_from_jacobi(mjac, Mjac)
        for mass, mass2 in zip(m,m2):
            self.assertAlmostEqual(mass, mass2)

    def test_actionangle(self):
        A=3.
        a=np.pi/4.
        X,Y = ActionAngleToXY(A,a)
        A2,a2 = XYToActionAngle(X,Y)
        self.assertAlmostEqual(A,A2)
        self.assertAlmostEqual(a,a2)

if __name__ == '__main__':
    unittest.main()

