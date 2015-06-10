// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

# include <iostream>
# include <fstream>

# include <functional>

# include <vector>
# include <memory>


# include "spline.hpp"

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


int main() {

	//np5_1::examples::evaluate_spline_1(std::cout);
	//np5_1::examples::evaluate_poly(std::cout);

	std::ofstream ofs("out.txt");
	np5_1::examples::evaluate_kernel(ofs);
	//np5_1::examples::run_optimization(std::cout);
	return 0;
}

