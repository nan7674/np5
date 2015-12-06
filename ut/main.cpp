# define BOOST_TEST_MODULE //sequence_test
# define BOOST_TEST_DYN_LINK
 
# include <vector>
# include <cmath>
# include <functional>
 
# include <boost/test/unit_test.hpp>

# include <iostream>

# include "mcore/sequence.hpp"
# include "mcore/calc.hpp"
# include "mcore/linalg.hpp"
# include "polynomial.hpp"

namespace {
	
	double const TOL = 1.e-10;
	
	template <typename T>
	bool eq(T x, T y, double tol=TOL) noexcept;
	
	template <>
	bool eq(double x, double y, double tol) noexcept {
		return std::fabs(x - y) < tol;
	}
	
	struct point {
		point(double xx, double yy) noexcept
			: x(xx), y(yy) {}

		double x;
		double y;	
	};
	
	void create_test_data(mcore::calc::polynom& p, std::vector<point>& ps) {
		mcore::calc::polynom P = {1, 2, 3};
		double dx = 0.02;
		double const xmax = 1;
	
		std::vector<point> qs;
		for (double x = 0; x < xmax; x += dx)
			qs.emplace_back(point{x, P(x)});

		std::swap(p, P);
		std::swap(ps, qs);
	}
	
	struct rosenbrock {
		rosenbrock(double aa, double bb) noexcept
			: a(aa), b(bb) {}
			
		double operator()(mcore::linalg::vec const& v) const noexcept {
			assert(v.dim() == 2);
			double const s1 = a - v(0);
			double const s2 = v(1) - v(0) * v(0);
			return s1 * s1 + b * s2 * s2;
		}	
		
		double a;
		double b;
	};

	
} // anonymous namespace


BOOST_AUTO_TEST_CASE(sequence_addition_test_1) {
	mcore::detail::sequence<int> S1 = {1, 2, 3};
	mcore::detail::sequence<int> S2 = {4, 5, 6};
	
	mcore::detail::sequence<int> s = mcore::detail::add<
		mcore::detail::sequence<int>, 
		mcore::detail::sequence<int>,
		mcore::detail::equal_size_policy>(S1, S2);
		
	BOOST_CHECK(s.size() == 3);
	BOOST_CHECK(s[0] == 1 + 4);
	BOOST_CHECK(s[1] == 2 + 5);
	BOOST_CHECK(s[2] == 3 + 6);
}

BOOST_AUTO_TEST_CASE(sequence_addition_test_2) {
	mcore::detail::sequence<int> S1 = {1, 2, 3};
	mcore::detail::sequence<int> S2 = {4, 5, 6, 7};
	
	mcore::detail::sequence<int> s = mcore::detail::add<
		mcore::detail::sequence<int>, 
		mcore::detail::sequence<int>,
		mcore::detail::indifferent_size_policy>(S1, S2);
		
	BOOST_CHECK(s.size() == 4);
	BOOST_CHECK(s[0] == 1 + 4);
	BOOST_CHECK(s[1] == 2 + 5);
	BOOST_CHECK(s[2] == 3 + 6);
	BOOST_CHECK(s[3] == 7);
}

BOOST_AUTO_TEST_CASE(sequence_addition_test_3) {
	mcore::detail::sequence<int> S1 = {1, 2, 3};
	mcore::detail::sequence<int> S2 = {4, 5, 6, 7};
	
	mcore::detail::sequence<int> s = mcore::detail::add<
		mcore::detail::sequence<int>, 
		mcore::detail::sequence<int>,
		mcore::detail::indifferent_size_policy>(S2, S1);
		
	BOOST_CHECK(s.size() == 4);
	BOOST_CHECK(s[0] == 1 + 4);
	BOOST_CHECK(s[1] == 2 + 5);
	BOOST_CHECK(s[2] == 3 + 6);
	BOOST_CHECK(s[3] == 7);
}

BOOST_AUTO_TEST_CASE(sequence_subtraction_test_1) {
	mcore::detail::sequence<int> S1 = {1, 2, 3};
	mcore::detail::sequence<int> S2 = {4, 5, 6};
	
	mcore::detail::sequence<int> s = mcore::detail::sub<
		mcore::detail::sequence<int>, 
		mcore::detail::sequence<int>,
		mcore::detail::equal_size_policy>(S1, S2);
		
	BOOST_CHECK(s.size() == 3);
	BOOST_CHECK(s[0] == 1 - 4);
	BOOST_CHECK(s[1] == 2 - 5);
	BOOST_CHECK(s[2] == 3 - 6);
}

BOOST_AUTO_TEST_CASE(sequence_subtraction_test_2) {
	mcore::detail::sequence<int> S1 = {1, 2, 3};
	mcore::detail::sequence<int> S2 = {4, 5, 6, 7};
	
	mcore::detail::sequence<int> s = mcore::detail::sub<
		mcore::detail::sequence<int>, 
		mcore::detail::sequence<int>,
		mcore::detail::indifferent_size_policy>(S1, S2);
		
	BOOST_CHECK(s.size() == 4);
	BOOST_CHECK(s[0] == 1 - 4);
	BOOST_CHECK(s[1] == 2 - 5);
	BOOST_CHECK(s[2] == 3 - 6);
	BOOST_CHECK(s[3] ==   - 7);
}

BOOST_AUTO_TEST_CASE(sequence_subtraction_test_3) {
	mcore::detail::sequence<int> S1 = {1, 2, 3};
	mcore::detail::sequence<int> S2 = {4, 5, 6, 7};
	
	mcore::detail::sequence<int> s = mcore::detail::sub<
		mcore::detail::sequence<int>, 
		mcore::detail::sequence<int>,
		mcore::detail::indifferent_size_policy>(S2, S1);
		
	BOOST_CHECK(s.size() == 4);
	BOOST_CHECK(s[0] == 4 - 1);
	BOOST_CHECK(s[1] == 5 - 2);
	BOOST_CHECK(s[2] == 6 - 3);
	BOOST_CHECK(s[3] == 7);
}

BOOST_AUTO_TEST_CASE(sequence_multiplication_1) {
	mcore::detail::sequence<int> S1 = {1, 2, 3};
	mcore::detail::sequence<int> s = mcore::detail::multiply(3, mcore::detail::multiply(S1, 3));
	assert(s.size() == 3);
	assert(s[0] == 1 * 3 * 3);
	assert(s[1] == 2 * 3 * 3);
	assert(s[2] == 3 * 3 * 3);
}

BOOST_AUTO_TEST_CASE(polynomial_test_const_1) {
	mcore::calc::polynom const H = {1, 1.5};
	BOOST_CHECK(H.degree() == 1);
	BOOST_CHECK(H[0] == 1);
	BOOST_CHECK(H[1] == 1.5);
	BOOST_CHECK(H[2] == 0);
}

BOOST_AUTO_TEST_CASE(polynomial_test_nonconst_1) {
	mcore::calc::polynom H = {1, 1.5};
	BOOST_CHECK(H.degree() == 1);
	
	BOOST_CHECK(H[0] == 1);
	BOOST_CHECK(H[1] == 1.5);
	
	H[2] = 0.5;
	
	BOOST_CHECK(H.degree() == 2);
	BOOST_CHECK(H[2] == 0.5);
}

BOOST_AUTO_TEST_CASE(polynomial_test_addition_1) {
	mcore::calc::polynom H1 = {1, 1.5};
	BOOST_CHECK(H1.degree() == 1);
	mcore::calc::polynom H2 = {0., 0, 1};
	BOOST_CHECK(H2.degree() == 2);
	
	mcore::calc::polynom S = H1 + H2;
	BOOST_CHECK(S.degree() == 2);
	
	BOOST_CHECK(S[0] == H1[0]);
	BOOST_CHECK(S[1] == H1[1]);
	BOOST_CHECK(S[2] == H2[2]);
}

BOOST_AUTO_TEST_CASE(polynomial_test_addition_2) {
	mcore::calc::polynom H1 = {1, 1.5};
	BOOST_CHECK(H1.degree() == 1);
	mcore::calc::polynom H2 = {0., 0, 1};
	BOOST_CHECK(H2.degree() == 2);
	
	mcore::calc::polynom S = 2 * (H1 + 1.5 * H2);
	BOOST_CHECK(S.degree() == 2);
	
	BOOST_CHECK(S[0] == (2 * H1[0]));
	BOOST_CHECK(S[1] == (2 * H1[1]));
	BOOST_CHECK(S[2] == (3 * H2[2]));
}

BOOST_AUTO_TEST_CASE(polynomial_test_addition_3) {
	mcore::calc::polynom H1 = {1, 1.5};
	BOOST_CHECK(H1.degree() == 1);
	mcore::calc::polynom H2 = {0., 0, 1};
	BOOST_CHECK(H2.degree() == 2);
	
	mcore::calc::polynom S = (H1 + 1.5 * H2) * 2;
	BOOST_CHECK(S.degree() == 2);
	
	BOOST_CHECK(S[0] == (2 * H1[0]));
	BOOST_CHECK(S[1] == (2 * H1[1]));
	BOOST_CHECK(S[2] == (3 * H2[2]));
}

BOOST_AUTO_TEST_CASE(polynomial_test_subtraction_1) {
	mcore::calc::polynom H1 = {1, 1.5};
	BOOST_CHECK(H1.degree() == 1);
	mcore::calc::polynom H2 = {0., 0, 1};
	BOOST_CHECK(H2.degree() == 2);
	
	mcore::calc::polynom S = H1 - H2;
	BOOST_CHECK(S.degree() == 2);
	
	BOOST_CHECK(S[0] == H1[0]);
	BOOST_CHECK(S[1] == H1[1]);
	BOOST_CHECK(S[2] == -H2[2]);
}

BOOST_AUTO_TEST_CASE(polynomial_test_subtraction_2) {
	mcore::calc::polynom H1 = {1, 1.5};
	BOOST_CHECK(H1.degree() == 1);
	mcore::calc::polynom H2 = {0., 0, 1};
	BOOST_CHECK(H2.degree() == 2);
	
	mcore::calc::polynom S = H2 - H1;
	BOOST_CHECK(S.degree() == 2);
	
	BOOST_CHECK(S[0] == -H1[0]);
	BOOST_CHECK(S[1] == -H1[1]);
	BOOST_CHECK(S[2] == H2[2]);
}

BOOST_AUTO_TEST_CASE(polynomial_test_subtraction_3) {
	mcore::calc::polynom H2 = {0., 0, 1};
	BOOST_CHECK(H2.degree() == 2);
	
	mcore::calc::polynom S = H2 - H2;
	BOOST_CHECK(S.degree() == 2);
	
	BOOST_CHECK(S[0] == 0);
	BOOST_CHECK(S[1] == 0);
	BOOST_CHECK(S[2] == 0);
}

BOOST_AUTO_TEST_CASE(polynomial_test_value) {
	mcore::calc::polynom H = {1, 1.5};
	BOOST_CHECK(H.degree() == 1);
	
	BOOST_CHECK(H[0] == 1);
	BOOST_CHECK(H[1] == 1.5);
	
	double const x = 2.3;
	double const fx = H(x);
	
	BOOST_CHECK(fx == (1 + 1.5 * x));
}

// =====================================================================
// Vector operations
// =====================================================================
BOOST_AUTO_TEST_CASE(vec_test_1) {
	mcore::linalg::vec v = {1, 2, 3};
	
	BOOST_CHECK(v.dim() == 3);
	BOOST_CHECK(v(0) == 1);
	BOOST_CHECK(v(1) == 2);
	BOOST_CHECK(v(2) == 3);
}

BOOST_AUTO_TEST_CASE(vec_increment_1) {
	mcore::linalg::vec v1 = {1, 2, 3};
	mcore::linalg::vec v2 = {4, 5, 6};
	
	BOOST_CHECK(v1.dim() == 3);
	BOOST_CHECK(v2.dim() == 3);
	
	v1 += v2;
	
	BOOST_CHECK(v1(0) == 1 + 4);
	BOOST_CHECK(v1(1) == 2 + 5);
	BOOST_CHECK(v1(2) == 3 + 6);
}

BOOST_AUTO_TEST_CASE(vec_decrement_1) {
	mcore::linalg::vec v1 = {1, 2, 3};
	mcore::linalg::vec v2 = {4, 5, 6};
	
	BOOST_CHECK(v1.dim() == 3);
	BOOST_CHECK(v2.dim() == 3);
	
	v1 -= v2;
	
	BOOST_CHECK(v1(0) == 1 - 4);
	BOOST_CHECK(v1(1) == 2 - 5);
	BOOST_CHECK(v1(2) == 3 - 6);
}

BOOST_AUTO_TEST_CASE(vec_mul_assign_1) {
	mcore::linalg::vec v = {1, 2, 3};
	BOOST_CHECK(v.dim() == 3);
	
	v *= 2;
	
	BOOST_CHECK(v.dim() == 3);
	BOOST_CHECK(v(0) == 1 * 2);
	BOOST_CHECK(v(1) == 2 * 2);
	BOOST_CHECK(v(2) == 3 * 2);
}

BOOST_AUTO_TEST_CASE(vec_div_assign_1) {
	mcore::linalg::vec v = {2, 4, 6};
	BOOST_CHECK(v.dim() == 3);
	
	v /= 2;
	
	BOOST_CHECK(v.dim() == 3);
	BOOST_CHECK(v(0) == 2. / 2.);
	BOOST_CHECK(v(1) == 4. / 2.);
	BOOST_CHECK(v(2) == 6. / 2.);
}

// Matrix testst
BOOST_AUTO_TEST_CASE(mat_test_1) {
	mcore::linalg::mat A(2, 2);
	A(0, 0) = 1;
	A(0, 1) = 0;
	A(1, 0) = 0;
	A(1, 1) = 1;
	
	BOOST_CHECK(A.rows() == 2);
	BOOST_CHECK(A.cols() == 2);
	
	BOOST_CHECK(A(0, 0) == 1);
	BOOST_CHECK(A(0, 1) == 0);
	BOOST_CHECK(A(1, 0) == 0);
	BOOST_CHECK(A(1, 1) == 1);
}

// Test for L2 approximation
BOOST_AUTO_TEST_CASE(L2_approximation_test) {
	mcore::calc::polynom ep;
	std::vector<point> ps;
	create_test_data(ep, ps);
	mcore::calc::polynom p = np5::approximate_l2(std::begin(ps), std::end(ps), 2);
	
	BOOST_CHECK(p.degree() == 2);
	BOOST_CHECK(eq(p[0], ep[0]));
	BOOST_CHECK(eq(p[1], ep[1]));
	BOOST_CHECK(eq(p[2], ep[2])); 
}

// Test for Nelder-Mead optimization
BOOST_AUTO_TEST_CASE(Nelder_Mead_optimization) {
	rosenbrock R(3, 100);
	mcore::linalg::vec initial = {-3, -4};
	mcore::calc::configuration_NM config;
	mcore::linalg::vec out = mcore::calc::optimize_NM(R, initial, config);
	
	BOOST_CHECK(out.dim() == 2);
	BOOST_CHECK(eq(R.a, out[0], 1.e-6));
	BOOST_CHECK(eq(R.a * R.a, out[1], 1.e-6));
}
