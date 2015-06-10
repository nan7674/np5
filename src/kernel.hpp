/** @brief Different approaches to the kernel regression
 *
 * http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/xlghtmlnode34.html
 *
 */
# pragma once

# include <cassert>
# include <functional>

# include "common.hpp"



namespace np5 {

	/** @brief Kernel for approximation
	 */
	struct qkernel {
		double operator()(double const x) const noexcept {
			if (x < -1)
				return 0;
			else if (x > 1)
				return 0;
			else {
				double const p = 1 - x * x;
				return p * p;
			}
		}
	};


	/*! @brief Analyses data in a container
	 *
	 * @param begin starting iterator over the container
	 * @param end   ending iterator
	 *
	 * @return pair of minimal and maximal distances
	 *
	 * The functions runs through the container and do two things:
	 *  - defines minimal distance betwwen two closed points;
	 *  - check if the container has been sorted.
	 */
	template <typename It>
	std::tuple<double, double> get_spreading(It begin, It end) {
		// find the maximal distance between two points
		It end_copy(end);
		std::advance(end_copy, -1);
		double const max_ws = end_copy->x - begin->x;

		// find minimal distance between points
		double min_ws = max_ws;
		double x0 = begin->x;
		++begin;
		for (; begin != end; ++begin) {
			double const x1 = begin->x;
			double const ws = x1 - x0;
			if (ws < 0)
				throw std::invalid_argument("The container is unsorted");

			if (ws < min_ws)
				min_ws = ws;
			x0 = x1;
		}
		return std::make_tuple(min_ws, max_ws);
	}


	/*! @brief Computes value of a KR on a data
	 *
	 * @param x point at which the value should be computed
	 * @param h bandwidth
	 * @param b starting iterator
	 * @param e end iterator
	 *
	 * @return computed value
	 */
	template <typename It>
	std::tuple<double, double> estimate_value(
			It b, It e, double const x, double const bandwidth) noexcept {
		double v = 0, q = 0;
		qkernel kernel;

		for (; b != e; ++b) {
			double const lw = kernel((x - b->x) / bandwidth);
			v += lw * b->y;
			q += lw;
		}

		return std::make_tuple(v, q);
	}


	/** @brief Computes value of a kernel regression
	 */
	template <typename It>
	double predict(It b, It e, double const x, double const bandwidth) noexcept {
		auto p = estimate_value(b, e, x, bandwidth);
		return std::get<0>(p) / std::get<1>(p);
	}


	/** @brief Given data points computes optimal bandwidth
	 */
	template <typename It>
	double compute_bandwidth(It pb, It pe) noexcept {
		auto bounds = get_spreading(pb, pe);

		double bandwidth = std::get<0>(bounds);
		double const max_bandwidth = std::get<1>(bounds);
		double const step = bandwidth * 0.5;

		It b = pb; std::advance(b, 1);
		It e = pe; std::advance(e, -1);

		double optimal_bandwidth = bandwidth;
		double min_error = std::numeric_limits<double>::max();

		for (; bandwidth < max_bandwidth; bandwidth += step) {
			double error_est = 0;
			for (It iter = b; iter != e; ++iter) {
				auto p0 = estimate_value(pb, iter, iter->x, bandwidth);
				auto p1 = estimate_value(iter + 1, pe, iter->x, bandwidth);

				double const V = (std::get<0>(p0) + std::get<0>(p1)) / (std::get<1>(p0) + std::get<1>(p1)) - iter->y;
				error_est += V * V;
			}

			if (error_est < min_error) {
				min_error = error_est;
				optimal_bandwidth = bandwidth;
			}
		}

		return optimal_bandwidth;
	}

	/** @brief Given a data and a local polynomial parameters computes a value of the regressor.
	 */
	template <typename It>
	double loc_poly_predict(It iter, It const iter_end, double const x, double const bandwidth) noexcept {
		double A = 0, B = 0, C = 0, D1 = 0, D2 = 0;
		qkernel kernel;
		for (; iter != iter_end; ++iter) {
			double const dx = x - iter->x;
			double const w = kernel(dx / bandwidth);

			A += w * dx * dx;
			B += w * dx;
			C += w;

			D1 += w * iter->y * dx;
			D2 += w * iter->y;
		}

		double const DET = A * C - B * B;
		double const b = (A * D2 - B * D1) / DET;

		return b;
	}
}
