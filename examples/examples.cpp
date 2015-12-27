# include "examples.hpp"


# include <iostream>

# include "src/polynomial.hpp"
# include "utils/data.hpp"



void np5::examples::l2_approximation() {
	mcore::calc::polynom ep = np5::utils::create_random_polynom(2);
	std::vector<point> vs = np5::utils::tabulate(ep, 0, 1, 0.05);

	mcore::calc::polynom p = np5::approximate_l2(std::begin(vs), std::end(vs), 2);

	std::cout << p[0] << ' ' << ep[0] << std::endl;
	std::cout << p[1] << ' ' << ep[1] << std::endl;
	std::cout << p[2] << ' ' << ep[2] << std::endl;
}


void np5::examples::run_ransac() {
	mcore::calc::polynom ep = np5::utils::create_random_polynom(1);
	std::vector<point> vs = np5::utils::tabulate(ep, 0, 1, 0.05);
	np5::utils::add_outliers(vs, 10);

	auto p = np5::approximate_RANSAC(std::begin(vs), std::end(vs), 1);

	auto p2 = np5::approximate_l1(std::begin(vs), std::end(vs), 1);

	for (auto& pt : vs)
		std::cout << pt.x << ' ' << pt.y << ' '
			<< ep(pt.x) << ' ' << p(pt.x) << ' ' << p2(pt.x) << std::endl;
}
