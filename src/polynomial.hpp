# pragma once

/**
 *
 * TODO :: add M-estimators
 * reference: http://research.microsoft.com/en-us/um/people/zhang/inria/publis/tutorial-estim/node24.html
 *
 */

# include <cmath>
# include <vector>
# include <random>

# include "common.hpp"
# include "mcore/optimization.hpp"
# include "mcore/linalg.hpp"
# include "mcore/calc.hpp"

namespace np5 {
	
	namespace detail {
		
		/** @brief Creates classical L2 data approximation
		 *
		 * @param begin  starting iterator of the data
		 * @param end    end iterator of the data to be approximated
		 * @param degree required degree of an approximation
		 *
		 * @return a vector with polynom coefficients
		 */
		template <typename It>
		mcore::linalg::vec approximate_l2(It begin, It end, size_t const degree) {
			size_t const num_samples = std::distance(begin, end);
	
			size_t const N = 2 * degree + 1;
			std::vector<double> t(N, 0);
			mcore::linalg::vec b(degree + 1, 0);
	
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
	
			mcore::linalg::mat A(degree + 1, degree + 1);
			for (size_t i = 0; i < degree + 1; ++i)
				for (size_t j = 0; j < degree + 1; ++j)
					A(i, j) = t[i + j];
	
			mcore::linalg::vec x = mcore::linalg::solve(A, b);
	
			mcore::linalg::vec approximator(degree + 1);
			for (size_t i = 0; i < degree + 1; ++i)
				approximator[i] = x(i);
			return approximator;
		}

	} // namespace detail

	/** @brief Functor to compute L1 error
	 */
	template <typename It>
	class l1_estimation {
	public:
		l1_estimation(It first, It last) noexcept
			: first_(first), last_(last) {}

		double operator()(mcore::linalg::vec const& p) const noexcept {
			double r = 0;
			auto const& cs = p.data();
			for (It iter = first_; iter != last_; ++iter)
				r += std::fabs(mcore::calc::poly_eval(cs, iter->x) - iter->y);
			return r;
		}

	private:
		l1_estimation(l1_estimation&) = delete;
		l1_estimation& operator=(l1_estimation&) = delete;

	private:
		It const first_;
		It const last_;
	};

	class configuration_l1 : public mcore::calc::configuration_NM {
	public:
		enum class initial_point {
			RANDOM,
			LSQ
		};

	public:
		configuration_l1() noexcept
			: mcore::calc::configuration_NM(), initial_{initial_point::LSQ} {}

		initial_point init_type() const noexcept {
			return initial_;
		}

	private:
		initial_point initial_;
	};
	
	inline mcore::linalg::vec get_random_vector(size_t degree) {
		std::mt19937 gen;
		mcore::linalg::vec v(degree);
		for (size_t i = 0; i < degree; ++i)
			v[i] = std::uniform_real_distribution<>{-1, 1}(gen);
		return v;
	}
	
	/** @brief Creates classical L2 data approximation
	 *
	 * @param begin  starting iterator of the data
	 * @param end    end iterator of the data to be approximated
	 * @param degree required degree of an approximation
	 *
	 * @return a vector with polynom coefficients
	 */
	template <typename It>
	mcore::calc::polynom approximate_l2(It begin, It end, size_t const degree) {
		return detail::approximate_l2(begin, end, degree).release();
	}

	/** @brief Given a data set approximates it with a polynom
	 */
	template <typename It>
	::mcore::calc::polynom approximate_l1(It begin, It end, size_t degree, configuration_l1 const& cnf=configuration_l1{}) {
		mcore::linalg::vec cs = mcore::calc::optimize_NM(
			l1_estimation<It>{begin, end},
			cnf.init_type() == configuration_l1::initial_point::RANDOM
				? get_random_vector(degree)
				: detail::approximate_l2(begin, end, degree),
			cnf);
		return cs.release();
	}


	/** @brief Approximates data by RANSAC
	 */
	template <typename It>
	mcore::calc::polynom approximate_RANSAC(It begin, It end, size_t degree) {
		return mcore::calc::polynom();
	}


}
