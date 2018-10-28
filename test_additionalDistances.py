import unittest

import numpy as np

from additionalDisctances import dist


class TestAdditionalDistances(unittest.TestCase):

    def test_type_pearson(self):
        self.assertRaises(TypeError, lambda: dist.pearson([0, 0, 0, 0], [0, 0, 0, 0]))
        self.assertRaises(TypeError, lambda: dist.pearson(-2, "M"))
        self.assertRaises(TypeError, lambda: dist.pearson(5 + 4j, 99.0))

    def test_type_manhattan(self):
        self.assertRaises(TypeError, lambda: dist.manhattanDistance([0, 0, 0, 0], [0, 0, 0, 0]))
        self.assertRaises(TypeError, lambda: dist.manhattanDistance(-2, "M"))
        self.assertRaises(TypeError, lambda: dist.manhattanDistance(5 + 4j, 99.0))

    def test_equal_pearson(self):
        self.assertEqual(dist.pearson(np.array([5.1, 3.5, 1.4, 0.2]), np.array([4.4, 3.2, 1.3, 0.2])),
                         0.0009078559637634155)
        self.assertEqual(dist.pearson(np.array([5.9, 3.0, 5.1, 1.8]
                                               ), np.array([6.5, 3.0, 5.2, 2.0])),
                         0.005539878843924153
                         )

    def test_equal_manhattan(self):
        self.assertEqual(dist.manhattanDistance(np.array([6.1, 2.9, 4.7, 1.4]
                                                         ), np.array([6.5, 3.0, 5.2, 2.0])), 1.6000000000000005)
        self.assertEqual(dist.manhattanDistance(np.array([5.6, 2.9, 3.6, 1.3]
                                                         ), np.array([6.5, 3.0, 5.2, 2.0]
                                                                     )), 3.3000000000000007
                         )
