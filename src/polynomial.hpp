# pragma once

/**
 *
 * TODO :: add M-estimators
 * reference: http://research.microsoft.com/en-us/um/people/zhang/inria/publis/tutorial-estim/node24.html
 *
 */

# include <type_traits>
# include <iterator>
# include <cmath>
# include <random>
# include <tuple>

# include <vector>

# include "common.hpp"
# include "mcore/optimization.hpp"
# include "mcore/linalg.hpp"
# include "mcore/calc.hpp"
# include "mcore/sequence.hpp"

namespace np5 {


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

	inline mcore::linalg::vec get_random_vector(size_t degree) {
		std::mt19937 gen;
		mcore::linalg::vec v(degree);
		for (size_t i = 0; i < degree; ++i)
			v[i] = std::uniform_real_distribution<>{-1, 1}(gen);
		return v;
	}


	namespace detail {

		class l2_calculator {
			l2_calculator(l2_calculator&) = delete;
			l2_calculator& operator=(l2_calculator&) = delete;

		public:
			l2_calculator(size_t degree)
				: degree_(degree),
				  solver_(degree + 1),
				  A_(degree + 1, degree + 1),
				  b_(degree + 1, 0),
				  work_(2 * degree + 1, 0) {}

			template <typename It>
			mcore::linalg::vec operator()(It begin, It end) {
				fill_data(begin, end);
				return solver_.solve(A_, b_);
			}

			template <typename It>
			void operator()(It begin, It end, mcore::linalg::vec& out) {
				assert(out.dim() == degree_ + 1);
				fill_data(begin, end);
				solver_.solve(A_, b_, out);
			}

		private:
			template <typename It>
			void fill_data(It begin, It end) {
				b_.to_zero();
				work_.to_zero();

				for (; begin != end; ++begin) {
					double e = 1;
					double const x = begin->x;
					for (size_t j = 0; j < degree_ + 1; ++j, e *= x) {
						work_[j] += e;
						b_(j) += begin->y * e;
					}
					for (size_t j = degree_ + 1; j < (2 * degree_ + 1); ++j, e *= x)
						work_[j] += e;
				}

				for (size_t i = 0; i < degree_ + 1; ++i)
					for (size_t j = 0; j < degree_ + 1; ++j)
						A_(i, j) = work_[i + j];
			}

		private:
			size_t degree_;
			mcore::linalg::leq_solver solver_;
			mcore::linalg::mat A_;
			mcore::linalg::vec b_;
			mcore::linalg::vec work_;
		};

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
			return l2_calculator(degree)(begin, end);
		}


		/** @brief Calculates L1 data aproximation
		 *
		 * @param begin initial position of data to be approximated
		 * @param end   end poisition of data to be approximated
		 * @param degree desired degree of approxmation
		 *
		 * @return vector with polynom coefficients
		 */
		template <typename It>
		mcore::linalg::vec approximate_l1(
			It begin,
			It end,
			size_t degree,
			configuration_l1 const& cnf=configuration_l1{})
		{
			return mcore::calc::optimize_NM(
				l1_estimation<It>{begin, end},
				cnf.init_type() == configuration_l1::initial_point::RANDOM
					? get_random_vector(degree)
					: detail::approximate_l2(begin, end, degree),
				cnf);
		}

		class num_generator {
			typedef mcore::detail::sequence<size_t> container_type;

			num_generator(num_generator&) = delete;
			num_generator& operator=(num_generator&) = delete;

		public:
			class random_sequence : public mcore::detail::data_view<size_t> {
				friend class num_generator;

				random_sequence(size_t* initial, size_t count) noexcept
					: mcore::detail::data_view<size_t>(initial, count) {}
			};

		public:
			/** @brief Creates num_generator object
			 *
			 * @param upper_bound upper bound of the numbers
			 * @param count       total number of samples to be generated
			 */
			num_generator(size_t upper_bound, size_t count);

			random_sequence generate() noexcept;

		private:
			void shuffle() noexcept;

		private:
			container_type data_;
			size_t const count_;
			size_t cursor_;

			std::random_device rd_;
			std::mt19937       generator_;
		};
	} // namespace detail

	/** @brief Computes R2 values
	 */
	template <typename It>
	double computeR2(It iter, It end, mcore::linalg::vec const& v) noexcept {
		double r2 = 0;
		size_t num_samples = 0;
		auto const& cs = v.data();
		for (; iter != end; ++iter, ++num_samples) {
			double const dx = iter->y - mcore::calc::poly_eval(cs, iter->x);
			r2 += dx * dx;
		}
		return r2 / double(num_samples);
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
	::mcore::calc::polynom approximate_l1(
		It begin, It end, size_t degree,
		configuration_l1 const& cnf=configuration_l1{})
	{
		return detail::approximate_l1(begin, end, degree, cnf).release();
	}

	struct ransac_conf {
		/** @brief Total number of iterations
		 */
		size_t num_iterations{500};

		/** @brief Number of samples to generate test models
		 */
		size_t num_samples{0};

		/** @brief Tolerance
		 */
		double tolerance{0.05};
	};

	/** @brief Approximates data by RANSAC
	 */
	template <typename It>
	mcore::calc::polynom approximate_RANSAC(
			It begin,
			It end,
			size_t degree,
			ransac_conf const& conf = ransac_conf()) {
		static_assert(
			std::is_same<
				typename std::iterator_traits<It>::iterator_category,
				std::random_access_iterator_tag>::value,
			"Current implementation supports only random access iterators");

		size_t const num_samples = std::distance(begin, end);
		std::vector<point> ps;
		ps.reserve(num_samples);

		size_t const TEST_DIM = conf.num_samples
			? conf.num_samples
			: degree + 1;

		if (TEST_DIM < degree + 1)
			throw std::runtime_error("Wrong number");

		mcore::linalg::vec best_model(degree + 1, 0);
		mcore::linalg::vec better_model(degree + 1, 0);
		mcore::linalg::vec model(degree + 1, 0);

		size_t max_set = 0;
		double R2 = std::numeric_limits<double>::max();

		detail::l2_calculator L2(degree);

		// Allocate memory for indexes
		detail::num_generator G(num_samples - 1, TEST_DIM);
		for (size_t i = 0; i < conf.num_iterations; ++i) {
			// Let's create a copy of the data
			for (size_t i : G.generate()) {
				assert((begin + i) < end);
				auto iter = begin + i;
				ps.emplace_back(point(iter->x, iter->y));
			}

			assert(ps.size() == degree + 1);
			L2(std::begin(ps), std::end(ps), model);
			ps.clear();

			// and select all the possible inliers
			for (It iter = begin; iter != end; ++iter) {
				assert(model.dim() == degree + 1);
				if (std::fabs(mcore::calc::poly_eval(model, iter->x) - iter->y) < conf.tolerance)
					ps.emplace_back(point(iter->x, iter->y));
			}

			if (ps.size() > max_set) {
				L2(std::begin(ps), std::end(ps), better_model);

				// Stupid verification for R2
				double const current_r2 = computeR2(std::begin(ps), std::end(ps), better_model);
				if (current_r2 < R2) {
					best_model.assign(better_model);
					max_set = ps.size();
					R2 = current_r2;
				}
			}

			ps.clear();
		}

		return best_model.release();
	}

}
