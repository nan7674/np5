import unittest
import numpy as np

import regression


TOL = 1.e-8


def build_data():
    T = (1, 0.5)

    def F(x):
        return T[0] + T[1] * x

    x, ds = 0, []
    while x < 1:
        ds.append((x, F(x)))
        x += 0.02
    return T, ds


def equal(x, y, tol=TOL):
    return abs(x - y) < tol


class test_l2_regression(unittest.TestCase):

    def test_tuple_data(self):
        'Tests tuple input.'
        t, data = build_data()
        self.assertTrue(isinstance(data, list))
        p = regression.approximate_l2(data, 1)
        self.assertTrue(equal(p[0], t[0]))
        self.assertTrue(equal(p[1], t[1]))

    def test_list_data(self):
        'Tests list input.'
        t, data = build_data()
        data = tuple(data)
        self.assertTrue(isinstance(data, tuple))
        p = regression.approximate_l2(data, 1)
        self.assertTrue(equal(p[0], t[0]))
        self.assertTrue(equal(p[1], t[1]))

    def test_wrong_input(self):
        with self.assertRaises(ValueError):
            d = {1: 2, 3: 4, 5: 6}
            regression.approximate_l2(d, 1)

    def test_insufficient_data(self):
        t, data = build_data()
        self.assertTrue(isinstance(data, list))
        with self.assertRaises(ValueError):
            regression.approximate_l2(data[:1], 1)

    def test_coefficients(self):
        rnd = np.random.uniform
        a, b, c = rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)
        F = np.vectorize(lambda x: a + b * x + c * x * x)

        xs = np.arange(0, 1, 0.02)
        ys = F(xs)

        p0 = np.polyfit(xs, ys, 2)
        fs = [(x, F(x)) for x in xs]
        p1 = np.array(regression.approximate_l2(fs, 2)[::-1])
        self.assertTrue(equal(sum(np.fabs(p0 - p1)), 0))


if __name__ == '__main__':
    unittest.main()
