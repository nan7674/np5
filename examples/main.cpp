# include <iostream>
# include <fstream>

# include <functional>

# include <vector>
# include <memory>


# include "src/spline.hpp"

# include <chrono>
/*! @brief Given the function runs it and measure time of execution
*
* @param[in] f function to run
* @return elapsed time in seconds
*/
double measure_time(std::function<void()> f) {
	auto start = std::chrono::system_clock::now();
	f();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);
	return duration.count() / 1000.;
}



# include <iostream>
# include <fstream>

# include "examples.hpp"

# include "src/polynomial.hpp"


int main() {
	//np5::examples::evaluate_spline_1(std::cout);
	np5::examples::evaluate_poly(std::cout);

	//np5::mcore::mat x;
	//np5::mcore::vec y;
	//np5::mcore::solve(x, y);

	//np5::examples::run_matrix(std::cout);

	//std::ofstream ofs("out.txt");
	//np5::examples::evaluate_kernel(std::cout);
	//np5::examples::run_optimization(std::cout);
	return 0;
}

