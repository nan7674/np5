# pragma once

# include "common.hpp"

# include <algorithm>
# include <cassert>
# include <functional>
# include <valarray>
# include <utility>


namespace np5 { namespace mcore {

	/*! @brief Shortcut for pow(x, 2)
	 */
	inline double sqr(double const x) noexcept {
		return x * x;
	}


	/*! @brief Given a function f and a point x computes derivative of the f
	 */
	template <typename F>
	double diff(F f, double const x) {
		static double const eps = 1.e-6;
		return (f(x + eps) - f(x - eps)) / (2. * eps);
	}


	/*! @brief Computes a second derivative of a given function
	 */
	template <typename F>
	double diff2(F f, double const x) {
		static double const eps = 1.e-6;
		static double const eps2 = eps * eps;
		double const fm = f(x - eps);
		double const f0 = f(x);
		double const fp = f(x + eps);
		return (fp + fm - 2 * f0) / eps2;
	}


	/** @brief Polynom abstraction
	 */
	typedef std::valarray<double> poly_type;


	/*! @brief Given a polynom and a point computes value of the polynom
	 *
	 * @param poly polynom which value is required
	 * @param x    a point
	 *
	 * The polynom is supposed to be written in the inverse order.
	 */
	inline double eval(poly_type const& poly, double const x) noexcept {
		size_t const sz = poly.size() - 1;
		double r = poly[sz];
		for (size_t i = sz - 1; i < sz; --i)
			r = r * x + poly[i];
		return r;
	}

	/** @brief Returns random poly
	 *
	 * @param deg  degree of the polynom
	 * @param cmin minimum of a coefficient
	 * @param cmax maximal value of a coefficient
	 *
	 * The function creates random polynom of the degree deg. Coefficients of
	 * the polynom is uniformly ditributed in the range [cmin, cmax]. By
	 * default cmin = -1, cmax = 1
	 */
	poly_type get_random_poly(
		size_t const degree,
		double const cmin=-1,
		double const cmax=1);


	struct conf_opt {
		conf_opt() noexcept
			: initial_step(0.01), min_step(1.e-6), acc_factor(2.0),
			  step_delta(2), num_iterations(500) {}

		double initial_step;
		double min_step;
		double acc_factor;
		double step_delta;
		size_t num_iterations;
	};


	/*! @brief Runs the neigbour search step in the HJ method
	*
	* @param f    function to optimize
	* @param p    point for investigation
	* @param step required value of the step
	*
	* @return pair of two elements
	*
	* The function performs the first stage in the HJ optimization
	* method. It returns a pair of two elements, the first of which
	* equals to true if the step has been successful. The second entry
	* in the pair contains the minimal value of f reached during the
	* step.
	*/
	template <typename F, typename P>
	inline std::pair<bool, double> neigbour_search(
			F f, P& p, double const step) noexcept {
		double e_0 = f(p);

		for (size_t i = 0; i < p.size(); ++i) {
			double const initial_value = p[i];

			p[i] += step;
			double const e_p = f(p);

			if (e_p < e_0)
				return std::make_pair(true, e_p);

			p[i] = initial_value - step;
			double const e_m = f(p);

			if (e_m < e_0)
				return std::make_pair(true, e_m);

			p[i] = initial_value;
		}

		return std::make_pair(false, e_0);
	}

	/** @brief Looks for function minimum near a point
	 */
	template <typename F, typename P>
	inline std::pair<bool, double> neigbour_search_total(
			F f, P& p, double const step) {
		bool min_found = false;
		P      min_point(p);
		double min_value = f(p);
		for (size_t i = 0; i < p.size(); ++i) {
			double const v = p[i];

			p[i] -= step;
			double const fm = f(p);
			if (fm < min_value) {
				min_point = p;
				min_value = fm;
				min_found = true;
			}

			p[i] = v + step;
			double const fp = f(p);
			if (fp < min_value) {
				min_point = p;
				min_value = fp;
				min_found = true;
			}

			p[i] = v;
		}

		p = min_point;
		return std::make_pair(min_found, min_value);
	}


	/** @brief Hook-Jeevse optimization procedure
	 */
	template <typename F, typename P>
	P optimize_hj(F func, P p0, conf_opt const& cnf) {
		double step = cnf.initial_step;
		double D = cnf.step_delta;

		P p2(p0), p1(p0);
		for (size_t i = 0; i < cnf.num_iterations; ++i) {
			double min_reached = 0;

			while (true) {
				auto sr = neigbour_search(func, p1, step);
				if (sr.first) {
					min_reached = sr.second;
					break;
				}
				D += cnf.step_delta;
				step = cnf.initial_step / D;
				if (step < cnf.min_step)
					return p1;
			}

			// pattern move step
			while (true) {
				for (size_t i = 0; i < p2.size(); ++i)
					p2[i] = p1[i] + cnf.acc_factor * (p1[i] - p0[i]);

				auto p = neigbour_search(func, p2, step);
				if (p.second < min_reached && p.first) {
					std::swap(p0, p1);
					std::swap(p1, p2);
					min_reached = p.second;
					D -= cnf.step_delta;
					step = cnf.initial_step / D;
				}
				else {
					p0 = p1;
					break;
				}
			}
		}

		return std::move(p0);
	}

}}
