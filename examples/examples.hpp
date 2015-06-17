# pragma once

# include <vector>
# include <functional>
# include <ostream>

namespace np5 { namespace examples {

	struct data_point {
		data_point(double xx, double yy) throw()
			: x(xx), y(yy) {}

		double x;
		double y;
	};

	/** @brief Tabulates function.
	 */
	std::vector<data_point> tabulate(std::function<double(double)> F, double xmin = 0, double const xmax = 1, double const dx = 0.02);

	/** @brief Applies perturbation to a data set.
	 */
	void perturbate(std::function<double(double)> error, std::vector<data_point>& v);

	void evaluate_spline_1(std::ostream& stream);

	void evaluate_kernel(std::ostream& stream);

	void evaluate_poly(std::ostream& stream);

	void run_optimization(std::ostream& stream);

	void run_matrix(std::ostream&);
}}
