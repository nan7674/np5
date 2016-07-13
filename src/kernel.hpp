/** @brief Different approaches to the kernel regression
 *
 * http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/xlghtmlnode34.html
 *
 */
# pragma once

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


/** @brief Kernel predictor type
 */
enum class kernel_predictor {
	NW,
	LOCAL_POLY
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

/** @brief Functor to compute kernel prediction
 */
template <kernel_predictor>
struct kernel_predictor_eval;

/** @brief Implementation of NW kernel regressor
 */
template <>
struct kernel_predictor_eval<kernel_predictor::NW> {

	typedef std::tuple<double, double> data_type;

	/** @brief Given an input computes NW parameters
	 */
	template <typename It>
	static data_type evaluate_params(It iter, It iter_end, double const x, double const bandwidth) noexcept {
		double v = 0, q = 0;
		qkernel kernel;

		for (; iter != iter_end; ++iter) {
			double const lw = kernel((x - iter->x) / bandwidth);
			v += lw * iter->y;
			q += lw;
		}

	return std::make_tuple(v, q);
	}

	/** @brief Estimate value
	 */
	static double estimate_value(data_type const& lp, data_type const& rp) noexcept {
		return (std::get<0>(lp) + std::get<0>(rp)) / (std::get<1>(lp) + std::get<1>(rp));
	}

	/** @brief Estimate value
	 */
	static double estimate_value(data_type const& pars) noexcept {
		return std::get<0>(pars) / std::get<1>(pars);
	}
};


/** @brief IMplementation of LP predictor routines
 */
template <>
struct kernel_predictor_eval<kernel_predictor::LOCAL_POLY> {

	typedef std::tuple<double, double, double, double, double> data_type;

	/** @brief Given an input computes NW parameters
	 */
	template <typename It>
	static data_type evaluate_params(It iter, It iter_end, double const x, double const bandwidth) noexcept {
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

		return std::make_tuple(A, B, C, D1, D2);
	}

	static double estimate_value(data_type const& lp, data_type const& rp) {
		double const A  = std::get<0>(lp) + std::get<0>(rp);
		double const B  = std::get<1>(lp) + std::get<1>(rp);
		double const C  = std::get<2>(lp) + std::get<2>(rp);
		double const D1 = std::get<3>(lp) + std::get<3>(rp);
		double const D2 = std::get<4>(lp) + std::get<4>(rp);

		double const DET = A * C - B * B;
		return (A * D2 - B * D1) / DET;
	}

	static double estimate_value(data_type const& pars) {
		double A, B, C, D1, D2;
		std::tie(A, B, C, D1, D2) = pars;

		double const DET = A * C - B * B;
		return (A * D2 - B * D1) / DET;
	}

};


/** @brief Computes value of a kernel regression
 */
template <typename It, kernel_predictor Tp>
double predict(It b, It e, double const x, double const bandwidth) noexcept {
	typedef kernel_predictor_eval<Tp> predictor_type;
	return predictor_type::estimate_value(predictor_type::evaluate_params(b, e, x, bandwidth));
}


/** @brief Given data points computes optimal bandwidth
 */
template <typename It, kernel_predictor Tp>
double compute_bandwidth(It pb, It pe) noexcept {
	typedef kernel_predictor_eval<Tp> predictor_type;

	auto bounds = get_spreading(pb, pe);

	double bandwidth = std::get<0>(bounds);
	double const max_bandwidth = std::get<1>(bounds);
	double const step = bandwidth * 0.5;

	It e = pe; std::advance(e, -1);

	double optimal_bandwidth = bandwidth;
	double min_error = std::numeric_limits<double>::max();

	for (; bandwidth < max_bandwidth; bandwidth += step) {
		double error_est = 0;
		for (It iter = pb; iter != e; ++iter) {
			auto p0 = predictor_type::evaluate_params(pb, iter, iter->x, bandwidth);
			auto p1 = predictor_type::evaluate_params(iter + 1, pe, iter->x, bandwidth);

			double const V = predictor_type::estimate_value(p0, p1) - iter->y;
			error_est += V * V;
		}

		if (error_est < min_error) {
			min_error = error_est;
			optimal_bandwidth = bandwidth;
		}
	}

	return optimal_bandwidth;
}

template <typename It>
double compute_band_nw(It pb, It pe) noexcept {
	return compute_bandwidth<It, kernel_predictor::NW>(pb, pe);
}

template <typename It>
double predict_nw(It b, It e, double x, double bw) noexcept {
	return predict<It, kernel_predictor::NW>(b, e, x, bw);
}

} // namespace np5
