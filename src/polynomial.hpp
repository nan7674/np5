/**
 *
 * TODO :: add M-estimators
 * reference: http://research.microsoft.com/en-us/um/people/zhang/inria/publis/tutorial-estim/node24.html
 *
 */


# pragma once


# include <cmath>
# include <vector>

# include "mcore.hpp"
# include "common.hpp"
# include "mcore/optimization.hpp"
# include "mcore/linalg.hpp"

namespace np5 {

	/** @brief Functor to compute L1 error
	 */
	template <typename It>
	class l1_estimation {
	public:
		l1_estimation(It first, It last) noexcept
			: first_(first), last_(last) {}

		double operator()(mcore::poly_type const& p) const noexcept {
			double r = 0;
			for (It iter = first_; iter != last_; ++iter)
				r += std::fabs(mcore::eval(p, iter->x) - iter->y);
			return r;
		}

	private:
		l1_estimation(l1_estimation&) = delete;
		l1_estimation& operator=(l1_estimation&) = delete;

	private:
		It const first_;
		It const last_;
	};


	/** @brief Creates classical L2 data approximation
	 */
	template <typename It>
	mcore::poly_type approximate_l2(It begin, It end, size_t const degree) {
		size_t const num_samples = std::distance(begin, end);

		size_t const N = 2 * degree + 1;
		std::vector<double> t(N, 0);
		::mcore::linalg::vec b(degree + 1);
		b.clear();

		for (; begin != end; ++begin) {
			double e = 1;
			double z = begin->y;
			double const x = begin->x;
			for (size_t j = 0; j < degree + 1; ++j, e *= x) {
				t[j] += e;
				b(j) += begin->y * e;
			}
			for (size_t j = degree + 1; j < N; ++j, e *= x)
				t[j] += e;

		}

		::mcore::linalg::mat A(degree + 1, degree + 1);
		for (size_t i = 0; i < degree + 1; ++i)
			for (size_t j = 0; j < degree + 1; ++j)
				A(i, j) = t[i + j];

		::mcore::linalg::vec x = ::mcore::linalg::solve(A, b);

		mcore::poly_type approximator(0., degree + 1);
		for (size_t i = 0; i < degree + 1; ++i)
			approximator[i] = x(i);
		return approximator;
	}


	class configuration_l1 : public opt::configuration_NM {
	public:
		enum class initial_point {
			RANDOM,
			LSQ
		};

	public:
		configuration_l1() noexcept
			: opt::configuration_NM(), initial_{initial_point::LSQ} {}

		initial_point init_type() const noexcept {
			return initial_;
		}

	private:
		initial_point initial_;
	};


	/** @brief Given a data set approximates it with a polynom
	 */
	template <typename It>
	mcore::poly_type approximate_l1(It begin, It end, size_t degree, configuration_l1 const& cnf=configuration_l1{}) {
		return opt::optimize_NM(
			l1_estimation<It>{begin, end},
			cnf.init_type() == configuration_l1::initial_point::RANDOM
				? mcore::get_random_poly(degree)
				: approximate_l2(begin, end, degree),
			cnf);
	}


	/** @brief Approximates data by RANSAC
	 */


}
