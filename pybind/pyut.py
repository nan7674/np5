import unittest
import numpy as np

import regression


TOL = 1.e-8


def build_data(degree, add_noise=False):
    rnd = np.random.uniform

    if add_noise:
        def noise():
            return rnd(-0.08, 0.08)
    else:
        def noise():
            return 0

    cs = np.random.uniform(-1, 1, (degree + 1,))
    F = np.vectorize(lambda x: np.polyval(cs, x) + noise())
    return cs, [(x, F(x)) for x in np.arange(0, 1, 0.02)]


def equal(x, y, tol=TOL):
    return abs(x - y) < tol


class test_l2_regression(unittest.TestCase):

    def test_tuple_data(self):
        'Tests tuple input.'
        t, data = build_data(1)
        self.assertTrue(isinstance(data, list))
        p = regression.approximate_l2(data, 1)[::-1]
        self.assertTrue(equal(p[0], t[0]))
        self.assertTrue(equal(p[1], t[1]))

    def test_list_data(self):
        'Tests list input.'
        t, data = build_data(1)
        data = tuple(data)
        self.assertTrue(isinstance(data, tuple))
        p = regression.approximate_l2(data, 1)[::-1]
        self.assertTrue(equal(p[0], t[0]))
        self.assertTrue(equal(p[1], t[1]))

    def test_wrong_input(self):
        with self.assertRaises(ValueError):
            d = {1: 2, 3: 4, 5: 6}
            regression.approximate_l2(d, 1)

    def test_insufficient_data(self):
        t, data = build_data(1)
        self.assertTrue(isinstance(data, list))
        with self.assertRaises(ValueError):
            regression.approximate_l2(data[:1], 1)

    def test_coefficients_1(self):
        # Tests result of an approximation.
        # In this test the input is just an exact
        # values of a polynom
        _, data = build_data(2)
        p0 = np.polyfit([x[0] for x in data], [x[1] for x in data], 2)
        p1 = np.array(regression.approximate_l2(data, 2)[::-1])
        self.assertTrue(equal(sum(np.fabs(p0 - p1)), 0))

    def test_coefficients_2(self):
        # Tests result of an approximation.
        # In this test the input is just an exact
        # values of a polynom
        _, data = build_data(2, True)
        xs = [x[0] for x in data]
        ys = [x[1] for x in data]
        p0 = np.polyfit(xs, ys, 2)
        p1 = np.array(regression.approximate_l2(xs, ys, 2)[::-1])
        self.assertTrue(equal(sum(np.fabs(p0 - p1)), 0))

    def test_robust(self):
        p0, data = build_data(1)
        p0 = np.array(p0)
        p1 = np.array(regression.approximate_l1(data, 1)[::-1])
        self.assertTrue(equal(sum(np.fabs(p0 - p1)), 0, 1e-4))


if __name__ == '__main__':
    unittest.main()
