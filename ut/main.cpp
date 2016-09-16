# define BOOST_TEST_MODULE //sequence_test
# define BOOST_TEST_DYN_LINK

# include <vector>
# include <cmath>
# include <functional>
# include <random>

# include <boost/test/unit_test.hpp>
# include <boost/log/trivial.hpp>

# include "mcore/sequence.hpp"
# include "mcore/calc.hpp"
# include "mcore/linalg.hpp"

# include "polynomial.hpp"
# include "spline.hpp"


# include "utils/data.hpp"

# include <cstdio>


namespace {

using np5::point;

using np5::utils::approx;
using np5::utils::set_default_tolerance;
using np5::utils::add_outliers;
using np5::utils::tabulate;


void create_test_data(
	::mcore::calc::polynom& P,
	std::vector<point>& vs,
	size_t degree=3)
{
	np5::utils::create_random_polynom(degree).swap(P);
	np5::utils::tabulate(P, 0, 1, 0.05).swap(vs);
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


class logger_impl {
public:
	logger_impl() : file_(fopen("ut.log", "w")) {}

	~logger_impl() {
		fflush(file_);
		fclose(file_);
	}

	logger_impl& operator<<(char const* s) {
		fprintf(file_, "%s", s);
		return *this;
	}

	logger_impl& operator<<(double v) {
		fprintf(file_, "%g", v);
		return *this;
	}

	logger_impl& operator<<(char ch) {
		fprintf(file_, "%c", ch);
		return *this;
	}

private:
	FILE* file_;
};

typedef np5::singleton<logger_impl> logger;

} // anonymous namespace

# define LOG logger::instance()


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

// =====================================================================
// Tests for matrix operations
// =====================================================================
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

BOOST_AUTO_TEST_CASE(matrix_mutliplication) {
	mcore::linalg::mat A(2, 3);
	A <<
		1, 2, 3,
		4, 5, 6;

	mcore::linalg::mat B(3, 1);
	B << 7, 8, 8;

	mcore::linalg::mat C = A * B;
	BOOST_CHECK(C.rows() == 2);
	BOOST_CHECK(C.cols() == 1);

	set_default_tolerance(1.e-10);
	BOOST_CHECK(approx(C(0, 0)) == (7 + 2 * 8 + 3 * 8));
	BOOST_CHECK(approx(C(1, 0)) == (4 * 7 + 5 * 8 + 6 * 8));
}

BOOST_AUTO_TEST_CASE(matrix_transposition) {
	mcore::linalg::mat A(2, 3);
	A <<
		1, 2, 3,
		4, 5, 6;

	BOOST_CHECK(A.rows() == 2);
	BOOST_CHECK(A.cols() == 3);

	auto const At = A.T();

	BOOST_CHECK(At.rows() == 3);
	BOOST_CHECK(At.cols() == 2);

	BOOST_CHECK(approx(At(0, 0)) == 1);
	BOOST_CHECK(approx(At(0, 1)) == 4);
	BOOST_CHECK(approx(At(1, 0)) == 2);
	BOOST_CHECK(approx(At(1, 1)) == 5);
	BOOST_CHECK(approx(At(2, 0)) == 3);
	BOOST_CHECK(approx(At(2, 1)) == 6);
}


// Test for L2 approximation
BOOST_AUTO_TEST_CASE(L2_approximation_test) {
	mcore::calc::polynom ep;
	std::vector<point> ps;

	create_test_data(ep, ps, 2);
	mcore::calc::polynom p = np5::approximate_l2(std::begin(ps), std::end(ps), 2);

	set_default_tolerance(1.e-10);
	BOOST_CHECK(p.degree() == 2);
	BOOST_CHECK(approx(p[0]) == ep[0]);
	BOOST_CHECK(approx(p[1]) == ep[1]);
	BOOST_CHECK(approx(p[2]) == ep[2]);
}

// Test for Nelder-Mead optimization
BOOST_AUTO_TEST_CASE(Nelder_Mead_optimization) {
	rosenbrock R(3, 100);
	mcore::linalg::vec initial = {-3, -4};
	mcore::calc::configuration_NM config;
	mcore::linalg::vec out = mcore::calc::optimize_NM(R, initial, config);

	BOOST_CHECK(out.dim() == 2);
	BOOST_CHECK(approx(R.a, 1.e-6) == out[0]);
	BOOST_CHECK(approx(R.a * R.a, 1.e-6) == out[1]);
}

// =====================================================================
// Tests for LEQ solvers
// =====================================================================
BOOST_AUTO_TEST_CASE(LEQ_solver) {
	mcore::linalg::mat A(3, 3);
	A(0, 0) = 1;  A(0, 1) = 4;  A(0, 2) = 9;
	A(1, 0) = 16; A(1, 1) = 25; A(1, 2) = 36;
	A(2, 0) = 49; A(2, 1) = 64; A(2, 2) = 81;

	mcore::linalg::vec v(3);
	v(0) = 1; v(1) = 0.75; v(2) = 0.5;

	auto const r = mcore::linalg::solve(A, v);
	BOOST_CHECK(std::abs(r[0] - 0.60416667) < 1.e-7);
	BOOST_CHECK(std::abs(r[1] + 1.16666667) < 1.e-7);
	BOOST_CHECK(std::abs(r[2] - 0.5625) < 1.e-7);
}


BOOST_AUTO_TEST_CASE(tridiagonal_solver) {
	std::array<double, 3> d0 = {10, 1, 1};
	std::array<double, 3> d1 = {2, 2, 2};
	std::array<double, 3> d2 = {1, 1, 10};
	std::array<double, 3> y = {3, 4, 3};

	mcore::linalg::solve_tridiagonal(
		d0.data(), d1.data(), d2.data(),
		y.data(),
		3);

	BOOST_CHECK(std::abs(y[0] - 1) < 1.e-10);
	BOOST_CHECK(std::abs(y[1] - 1) < 1.e-10);
	BOOST_CHECK(std::abs(y[2] - 1) < 1.e-10);
}


// =============================================================================
// Tests for spline interpolation
// =============================================================================
namespace {

void run_spline_1(
	np5::spline_builder& factory,
	double const w0,
	double const w1,
	double const w2)
{
	std::function<double(double)> F =
		[w0, w1, w2](double x) { return w0 + x * w1 + x * x * w2; };
	std::function<double(double)> dF =
		[w1, w2](double x) { return w1 + 2 * w2 * x; };

	double const xs[] = {1, 1.1, 2.5, 2.7, 6.7, 17.8};
	std::vector<np5::point> ps;
	for (auto x : xs)
		ps.emplace_back(x, F(x));

	auto const S = factory(ps.begin(), ps.end(),
		np5::d1_boundary(dF(ps.front().x), dF(ps.back().x)));

	for (auto const& rec: S.nodes()) {
		auto const ws = rec.get_global();
		BOOST_CHECK(std::abs(std::get<0>(ws) - w0) < 1.e-10);
		BOOST_CHECK(std::abs(std::get<1>(ws) - w1) < 1.e-10);
		BOOST_CHECK(std::abs(std::get<2>(ws) - w2) < 1.e-10);
		BOOST_CHECK(std::abs(std::get<3>(ws)) < 1.e-10);
	}
}

void run_spline_2(
	np5::spline_builder& factory,
	double const w0,
	double const w1,
	double const w2)
{
	std::function<double(double)> F =
		[w0, w1, w2](double x) { return w0 + x * w1 + x * x * w2; };
	std::function<double(double)> d2F =
		[w2](double x) { return 2 * w2; };

	double const xs[] = {1, 1.1, 2.5, 2.7, 6.7, 17.8};
	std::vector<np5::point> ps;
	for (auto x : xs)
		ps.emplace_back(x, F(x));

	auto const S = factory(ps.begin(), ps.end(),
		np5::d2_boundary(d2F(ps.front().x), d2F(ps.back().x)));

	for (auto const& rec: S.nodes()) {
		auto const ws = rec.get_global();
		BOOST_CHECK(std::abs(std::get<0>(ws) - w0) < 1.e-10);
		BOOST_CHECK(std::abs(std::get<1>(ws) - w1) < 1.e-10);
		BOOST_CHECK(std::abs(std::get<2>(ws) - w2) < 1.e-10);
		BOOST_CHECK(std::abs(std::get<3>(ws)) < 1.e-10);
	}
}

} // anonymous namespace


BOOST_AUTO_TEST_CASE(spline_interpolation_data) {
	std::vector<np5::point> pts = {
		{1, 1},
		{2, 4},
		{3, 9},
		{3, 16},
		{5, 25}
	};

	np5::spline_builder factory;
	BOOST_CHECK_THROW(
		factory(pts.begin(), pts.end()),
		std::runtime_error);
}


BOOST_AUTO_TEST_CASE(spline_interpolation_d1) {
	np5::spline_builder factory;
	run_spline_1(factory, 1, 0, 0);
	run_spline_1(factory, 0, 1, 0);
	run_spline_1(factory, 0, 0, 1);
	run_spline_1(factory, 2.3, -4.5, 0.178);
}


BOOST_AUTO_TEST_CASE(spline_interpolation_d2) {
	np5::spline_builder factory;
	run_spline_2(factory, 1, 0, 0);
	run_spline_2(factory, 0, 1, 0);
	run_spline_2(factory, 0, 0, 1);
	run_spline_2(factory, 2.3, -4.5, 0.178);
}


BOOST_AUTO_TEST_CASE(spline_interpolation_at_sin) {
	std::function<double(double)> F = [](double x) { return sin(4 * x); };
	std::function<double(double)> dF = [](double x) { return 4 * cos(4 * x);};

	auto data0 = tabulate(F, -1, 1.1, 0.25);
	double const u0 = data0.front().x;
	double const un = data0.back().x;

	np5::spline_builder factory;
	auto const S = factory(data0.begin(), data0.end(),
		np5::d1_boundary(dF(u0), dF(un)));

	auto data1 = tabulate(F, -1, 1, 0.0001);

	double diff = 0;
	for (auto const& pt: data1) {
		double const d = std::abs(S(pt.x) - pt.y);
		if (d > diff)
			diff = d;
	}

	// Theoretical error estimation (not optimal but usable)
	double const theor_error = 1. / 16. * 256 * pow(0.25, 4);
	BOOST_CHECK(diff < theor_error);

}


// =============================================================================
// RANSAC approximation
// =============================================================================
BOOST_AUTO_TEST_CASE(RANSAC_test_1) {
	size_t const UPPER_BOUND = 30;
	np5::detail::num_generator G(UPPER_BOUND, 3);

	size_t const NUM_CYCLES = 2000;
	std::vector<double> P(UPPER_BOUND + 1);
	for (size_t i = 0; i < NUM_CYCLES; ++i) {
		auto const& d = G.generate();
		for (size_t j = 0; j < 3; ++j)
			P.at(d[j]) += 1;
	}
}


BOOST_AUTO_TEST_CASE(RANSAC_test_2) {
	mcore::calc::polynom ep;
	std::vector<point> ps;
	create_test_data(ep, ps, 2);

	ps[2].y += 3;
	auto p = np5::approximate_RANSAC(std::begin(ps), std::end(ps), 2);

	BOOST_CHECK(approx(ep[0]) == p[0]);
	BOOST_CHECK(approx(ep[1]) == p[1]);
	BOOST_CHECK(approx(ep[2]) == p[2]);
}


//BOOST_AUTO_TEST_CASE(RANSAC_test_3) {
//	mcore::calc::polynom ep;
//	std::vector<point> ps;
//	create_test_data(ep, ps, 1);

//	add_outliers(ps, 10);
//	np5::ransac_conf cnf;
//	cnf.num_iterations = 1000;
//	cnf.tolerance = 0.05;
//	cnf.num_samples = 3;
//	auto p = np5::approximate_RANSAC(std::begin(ps), std::end(ps), 1, cnf);

//	double error = 0;
//	for (auto const& pt: ps) {
//		double const ex = std::fabs(ep(pt.x) - p(pt.x));
//		if (ex > error)
//			error = ex;
//	}

//	LOG << "RANSAC 3 :\n";
//	LOG << "exact polynom : " << ep[0] << " + " << ep[1] << "*x" << '\n';
//	LOG << "computed polynom : " << p[0] << " + " << p[1] << "*x" << '\n';

//	double const d0 = std::fabs(ep[0] - p[0]);
//	double const d1 = std::fabs(ep[1] - p[1]);
//	LOG << "difference in coefficients : " << "d0=" << d0 << " d1=" << d1 << "; delta = " << error << '\n';

//	BOOST_CHECK(error < 0.1);
//}

# undef LOG
