# include "examples.hpp"

# include <iostream>
# include <iomanip>

# include <random>

# include "src/spline.hpp"
# include "src/mcore.hpp"
# include "src/kernel.hpp"
# include "src/polynomial.hpp"


namespace ex = np5::examples;

namespace {

	struct P {
		P(double aa, double bb, double cc, double dd) throw()
			: a(aa), b(bb), c(cc), d(dd) {}

		double operator()(double x) const throw() {
			return a + x * (b + x * (c + d * x));
		}

		double const a;
		double const b;
		double const c;
		double const d;
	};

	class generator{
	public:
		generator() {
			std::random_device rd;
			generator_.seed(rd());
		}

		std::mt19937& operator()() { return generator_; }

		static generator& instance() {
			static generator G;
			return G;
		}

	private:
		std::mt19937 generator_;
	};


	double random_double(double x, double y) {
		return std::uniform_real_distribution<>(x, y)(generator::instance()());
	}

} // anonymous namespace

std::vector<ex::data_point> ex::tabulate(std::function<double(double)> F, double xmin, double const xmax, double const dx) {
	std::vector<data_point> out;

	for (; xmin < xmax; xmin += dx)
		out.emplace_back(data_point(xmin, F(xmin)));

	return std::move(out);
}


void ex::perturbate(std::function<double(double)> error, std::vector<data_point>& v) {
	for (auto& rec : v)
		rec.y += error(rec.x);
}



void ex::evaluate_spline_1(std::ostream& stream) {
	double const dx = 0.02;
	double const xmin = 0;
	double const xmax = 5. + 0.5 * dx;

	P poly(1.234, -2.678, 3.456, -0.01);
	auto points = tabulate(poly, xmin, xmax, dx);

	np5::spline_builder sb;

	auto s1 = sb(std::begin(points), std::end(points));
	auto s2 = sb(std::begin(points), std::end(points), 0.01);

	double const ddx = 0.357 * dx;
	double const dxmax = xmax + 0.5 * ddx;

	for (double x = xmin; x < dxmax; x += ddx)
		stream << x << ' ' << poly(x) << ' ' << s1(x) << ' ' << s2(x) << '\n';
}


void ex::evaluate_poly(std::ostream& stream) {
	double const dx = 0.02;
	double const xmin = 0;
	double const xmax = 1. + 0.5 * dx;

	P poly(-1., 0.345, 5.64, 0);
	auto points = tabulate(poly, xmin, xmax, dx);
	auto data = points;

	std::function<double(double)> erf = [](double) { return random_double(-0.1, 0.1); };
	np5::examples::perturbate(erf, data);

	// Run L1 optimization
	auto p = np5::approximate_l1(std::begin(data), std::end(data), 2);

	for (size_t i = 0; i < p.size(); ++i)
		std::cout << p[i] << ' ';
	std::cout << std::endl;

	//np5::L1L2 fair(0.01);
	//auto gr = np5::make_grad_eval(data.begin(), data.end(), fair);

	//auto pt = np5::mcore::get_random_poly(3);

	//auto v = gr.gradient(pt);

	//std::function<double(size_t, double, np5::mcore::poly_type)> F = [&gr](size_t i, double x, np5::mcore::poly_type pt) {
	//	pt[i] = x;
	//	return gr.distance(pt);
	//};

	//for (size_t i = 0; i < pt.size(); ++i)
	//	std::cout << v[i] << ' ' << np5::mcore::diff([i, &F, &pt](double x) { return F(i, x, pt); }, pt[i]) << ' ' << std::endl;

	//auto J = gr.jacobian(pt);
	//for (size_t i = 0; i < pt.size(); ++i)
	//	for (size_t j = 0; j < pt.size(); ++j) {
	//		std::cout << J(i, j) << std::endl;
	//	}


	//np5::L1L2 m(0.1);
	//auto p = np5::approximate(std::begin(data), std::end(data), 3, m);

	//stream << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << p[3] << '\n';
}


void ex::evaluate_kernel(std::ostream& stream) {
	double const dx = 0.02;
	double const xmin = 0;
	double const xmax = 1. + 0.5 * dx;

	P poly(-1., 0.345, 2.64, 8.78);
	std::function<double(double)> F = (double(*)(double))sin;
	auto points = tabulate(F, xmin, xmax, dx);
	auto data = points;

	std::function<double(double)> erf = [](double) { return random_double(-0.1, 0.1); };
	np5::examples::perturbate(erf, data);

	double const ddx = 0.357 * dx;
	double const dxmax = xmax + 0.5 * ddx;

	double const bandwidth = np5::compute_bandwidth<decltype(data.begin()), np5::kernel_predictor::NW>(std::begin(data), std::end(data));

	for (auto const& r : data) {
		double const x = r.x;
		double const y = r.y;

		double const krn = np5::predict<decltype(points.begin()), np5::kernel_predictor::NW>(std::begin(points), std::end(points), x, bandwidth);
		double const lp  = np5::predict<decltype(points.begin()), np5::kernel_predictor::LOCAL_POLY>(std::begin(points), std::end(points), x, bandwidth);

		stream <<
			x << ' ' <<
			y << ' ' <<
			F(x) << ' ' <<
			krn << ' ' <<
			lp << '\n';
	}

}


void ex::run_optimization(std::ostream& stream) {

	std::function<double(std::valarray<double> const& p)> F =
		[](std::valarray<double> const& p) { return p[0] * p[0] + p[1] * p[1]; };

	std::valarray<double> x0(2);
	x0[0] = -0.67;
	x0[1] = 567;

	np5::mcore::conf_opt co;
	co.num_iterations = 500;
	co.initial_step = 1;
	co.min_step = 1.e-6;

	auto pt = np5::mcore::optimize_hj(F, x0, co);

	stream << pt[0] << ' ' << pt[1] << std::endl;
}


void ex::run_matrix(std::ostream& stream) {
	np5::mcore::mat A(5, 5);
	np5::mcore::mat I = np5::mcore::mat::idendity(5, 5);

	A += I;

	for (size_t i = 0; i < 5; ++i) {
		for (size_t j = 0; j < 5; ++j)
			stream << A(i, j) << ' ';
		std::cout << std::endl;
	}
}
