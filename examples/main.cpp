
# include <iostream>

# include "examples.hpp"
# include "utils/data.hpp"


int main() {
//	np5::examples::run_ransac();
//	np5::examples::run_nonparametric();


	double const et = np5::utils::measure_time(np5::examples::run_spline);
	std::cout << "Elapsed time:" << et << std::endl;

	return 0;
}

