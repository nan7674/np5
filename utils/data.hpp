# pragma once

# include <limits>
# include <functional>

# include "mcore/calc.hpp"
# include "common.hpp"

namespace np5 { namespace utils {

// Creates random poly of a given degree
mcore::calc::polynom create_random_polynom(size_t degree);

/*! \brief Tabulates a function over on interval
 *
 * @param f function to be tabulated
 * @param s initial point of the interval
 * @param e end point of the interval
 * @pram step step that will be used during tabulation
 */
template <typename F>
std::vector<np5::point> tabulate(
	F const& f, 
	double s, 
	double const e, 
	double const step)
{
	size_t const cap = (e - s) / step;
	std::vector<np5::point> out;
	out.reserve(cap);

	for (; s < e; s += step)
		out.emplace_back(s, f(s));

	return out;
}

/*! \brief Adds some outliers to a data
 *
 * @param data to perturbate
 * @param nouts number of required outliers
 *
 */
void add_outliers(std::vector<np5::point>& data, size_t nouts);


/*! \brief Sets a tolerance
 */
void set_default_tolerance(double) noexcept;


/*! \brief Class for double comparation
 */
class approx {
public:
	/*! \brief Ctor
	 *
	 * @param value a value to be compared
	 *
	 * During comparison the default (global) tolerance
	 * will be used
	 */
	approx(double value) noexcept
		: value_(value),
		  tolerance_(std::numeric_limits<double>::quiet_NaN()) {}

	/*! \brief Ctor
	 *
	 * @param value a value to comprare
	 * @param tolerance a value to be used during comparison
	 */
	approx(double value, double tolerance)
		: value_(value), tolerance_(tolerance) {}

	bool operator==(double other) const noexcept;

private:
	double value_;
	double tolerance_;
};


/*! @brief Measures time execution of a function
 *
 * @param[in] f function to executed
 * @return elapsed time in seconds
 */
double measure_time(std::function<void()> f);


}} // utils //np5
