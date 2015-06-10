# include "examples.hpp"

# include <iostream>
# include <iomanip>

# include <random>

# include "spline.hpp"
# include "mcore.hpp"
# include "kernel.hpp"
# include "polynomial.hpp"


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

}

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
	double const xmax = 1. + 0.5 * dx;

	P poly(1.234, -2.678, 3.456, -0.01);
	auto points = tabulate(poly, xmin, xmax, dx);

	np5::spline_builder sb;

	auto s1 = sb(std::begin(points), std::end(points));
	auto s2 = sb(std::begin(points), std::end(points), 0.);

	double const ddx = 0.357 * dx;
	double const dxmax = xmax + 0.5 * ddx;

	for (double x = xmin; x < dxmax; x += ddx)
		stream << x << ' ' << poly(x) << ' ' << s1(x) << ' ' << s2(x) << '\n';

	// Check data approximation in the center of the range
}


void ex::evaluate_poly(std::ostream& stream) {
	double const dx = 0.02;
	double const xmin = 0;
	double const xmax = 1. + 0.5 * dx;

	P poly(-1., 0.345, 5.64, 0);
	auto points = tabulate(poly, xmin, xmax, dx);
	auto data = points;

	std::function<double(double)> erf = [](double) { return random_double(-0.005, 0.005); };
	np5::examples::perturbate(erf, data);

	auto p = np5::approximate_l1(std::begin(data), std::end(data), 2);

	stream << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << p[3] << '\n';
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

	double const bandwidth = np5::compute_bandwidth(std::begin(data), std::end(data));

	for (auto const& r : data) {
		double const x = r.x;
		double const y = r.y;

		double const krn = np5::predict(std::begin(points), std::end(points), x, bandwidth);
		double const lp  = np5::loc_poly_predict(std::begin(points), std::end(points), x, bandwidth);

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
