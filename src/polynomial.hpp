/**
 * reference: http://research.microsoft.com/en-us/um/people/zhang/inria/publis/tutorial-estim/node24.html
 */


# pragma once


# include <cmath>

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


	/** @brief Given a data set approximates it with a polynom
	 */
	template <typename It>
	mcore::poly_type approximate_l1(It begin, It end, size_t const degree) {
		return opt::optimize_NM(
			l1_estimation<It>{begin, end},
			mcore::get_random_poly(degree),
			opt::configuration_NM{});
	}

	/** @brief Creates classical L2 data approximation
	 */
	template <typename It>
	mcore::poly_type approximate_l2(It begin, It end, size_t const degree) {
		size_t const num_samples = std::distance(begin, end);

		::mcore::linalg::mat A(num_samples, degree + 1);
		::mcore::linalg::vec b(num_samples);

		size_t r = 0;
		for (; begin != end; ++begin, ++r) {
			double e = 1;
			for (size_t j = 0; j < degree + 1; ++j, e *= begin->x)
				A(r, j) = e;
			b(r) = begin->y;
		}

		::mcore::linalg::vec x = ::mcore::linalg::solve(
			::mcore::linalg::transposed(A) * A,
			::mcore::linalg::transposed(A) * b);

		mcore::poly_type approximator(0., degree + 1);
		for (size_t i = 0; i < degree + 1; ++i)
			approximator[i] = x(i);
		return approximator;

	}


	// Approximation L1 norm with different functions

	class L1L2 {
	public:
		L1L2(double const spreading) noexcept
			: spreading_(spreading) {}


		double value(double const X) const noexcept {
			double const x = X / spreading_;
			return 2 * (std::sqrt(1 + x * x / 2) - 1);
		}

		double diff(double const X) const noexcept {
			double const x = X / spreading_;
			return x / std::sqrt(1 + x * x / 2) / spreading_;
		}

		double diff2(double const X) const noexcept {
			double const x = X / spreading_;
			return 1. / std::sqrt(1 + x * x / 2.);
		}

	private:
		double spreading_;
	};


	class Fair {
	public:
		Fair(double const spreading) noexcept
			: c_(spreading) {}

		double value(double const X) const noexcept {
			double const x = std::abs(X) / c_;
			return c_ * c_ * (x - std::log(1 + x));
		}

		double diff(double const x) const noexcept {
			return x / (1 + std::abs(x) / c_);
		}

	private:
		double c_;
	};


	/** @brief Functor to evaluate gradient
	 */
	template <typename It, typename W>
	class grad_eval {
		typedef mcore::poly_type argument_type;
		typedef mcore::vec       gradient_type;
		typedef mcore::mat       jacobian_type;

	public:
		/** Ctor
		 *
		 * @param iter      iterator pointed to the first element of the range
		 * @param iter_end  last element of the range
		 * @param measure   const reference to a weight function
		 */
		grad_eval(It iter, It iter_end, W const& measure) noexcept
			: begin_(iter), end_(iter_end), measure_(measure) {}

		/** @brief Measures discrepancy of the poly from the data
		 */
		double distance(argument_type const& poly) const noexcept {
			size_t const dim = poly.size();

			double r = 0;
			for (It iter = begin_; iter != end_; ++iter)
				r += measure_.value(mcore::eval(poly, iter->x) - iter->y);

			return r;
		}

		/** @brief Computes gradient of a poly
		 *
		 * @param poly a point at which gradient is required
		 * @return computed gradient
		 */
		gradient_type gradient(argument_type const& poly) const {
			size_t const dim = poly.size();
			gradient_type g(dim);

			for (It iter = begin_; iter != end_; ++iter) {
				double const x = iter->x;
				double w = measure_.diff(mcore::eval(poly, x) - iter->y);
				for (size_t j = 0; j < dim; ++j, w *= x)
					g[j] += w;
			}

			return g;
		}

		/** @brief Computes jacobian
		 */
		jacobian_type jacobian(argument_type const& poly) const {
			jacobian_type J(poly.size(), poly.size());

			for (It iter = begin_; iter != end_; ++iter)
				update_jacobian(J, poly, iter->x, iter->y);
			return J;
		}

	private:

		void update_jacobian(jacobian_type& J, argument_type const& poly, double const x, double const y) const noexcept {
			double const w = measure_.diff2(mcore::eval(poly, x) - y);
			double X = 1;
			for (size_t i = 0; i < poly.size(); ++i, X *= x) {
				double X1 = X;
				for (size_t j = 0; j < poly.size(); ++j, X1 *= x)
					J(i, j) += w * X1;
			}
		}

	private:
		It begin_;
		It end_;
		W const& measure_;
	};


	template <typename It, typename W>
	grad_eval<It, W> make_grad_eval(It b, It e, W const& measure) noexcept {
		return grad_eval<It, W>(b, e, measure);
	}


	template <typename It, class W>
	mcore::poly_type approximate(It iter, It iter_end, size_t const degree, W measure) {
		static_assert(np5::common::has_member_diff<W>::value,
			"Measure has to be differentiable function");
		static_assert(np5::common::has_member_diff2<W>::value,
			"Measure entity should have a method to compute second derivative");
		mcore::poly_type poly(degree + 1);

		return poly;
	}

}
