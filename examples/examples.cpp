# include "examples.hpp"


# include <cmath>
# include <iostream>

# include "src/polynomial.hpp"
# include "src/kernel.hpp"
# include "src/spline.hpp"
# include "utils/data.hpp"


namespace {

void print_header(std::ostream& stream, std::string const& name) {
	stream << "**********************************************\n";
	stream << "*** " << name << '\n';
	stream << "**********************************************\n";
}

void print_poly(
	std::ostream& stream,
	std::string const& hdr,
	mcore::calc::polynom const& p)
{
	stream << hdr << ':';
	for (size_t i = 0; i < p.degree(); ++i)
		stream << p[i] << ' ';
	stream << p[p.degree()] << std::endl;
}

using np5::utils::tabulate;

} // anonyous namespace


void np5::examples::l2_approximation() {
	mcore::calc::polynom ep = np5::utils::create_random_polynom(2);
	std::vector<point> vs = tabulate(ep, 0, 1, 0.05);

	mcore::calc::polynom p = np5::approximate_l2(std::begin(vs), std::end(vs), 2);

	std::cout << p[0] << ' ' << ep[0] << std::endl;
	std::cout << p[1] << ' ' << ep[1] << std::endl;
	std::cout << p[2] << ' ' << ep[2] << std::endl;
}


void np5::examples::run_ransac() {
	print_header(std::cout, "RANSAC && L1 approximation");
	mcore::calc::polynom ep = np5::utils::create_random_polynom(1);
	std::vector<point> vs = tabulate(ep, 0, 1, 0.05);

	np5::utils::add_outliers(vs, 10);

	auto p = np5::approximate_RANSAC(std::begin(vs), std::end(vs), 1);
	auto p2 = np5::approximate_l1(std::begin(vs), std::end(vs), 1);

	print_poly(std::cout, "Initial polynom", ep);
	print_poly(std::cout, "L1 approximation", p);
	print_poly(std::cout, "RANSAC", p2);

	for (auto& pt : vs)
		std::cout << pt.x << ' ' << pt.y << ' '
			<< ep(pt.x) << ' ' << p(pt.x) << ' ' << p2(pt.x) << std::endl;
}

void np5::examples::run_nonparametric() {
	print_header(std::cout, "Non-paramteric regression");

	// Generate test data
	double const step = 0.2;
	double const ini = 0;
	double const stop = 8;
	std::vector<point> vs = tabulate(sin, ini, stop, step);
	double const nw_bw = np5::compute_band_nw(vs.begin(), vs.end());
	std::cout << "NW predictor, bandwidth : "  << nw_bw << std::endl;

	// Print a data predcition
	for (double x = 0.5 * step + ini; x < stop; x += step) {
		double const ev = sin(x);
		double const av = np5::predict_nw(vs.begin(), vs.end(), x, nw_bw);
		std::cout << x << ' ' << ev << ' ' << av << ' ' << std::abs(ev - av) << '\n';
	}
}


void np5::examples::run_spline() {
	print_header(std::cout, "Run spline interpolation");
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
	std::cout << diff << ' ' << theor_error << std::endl;
}
