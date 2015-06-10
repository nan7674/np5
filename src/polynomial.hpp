# pragma once

# include <cmath>
# include <valarray>

# include "common.hpp"
# include "mcore.hpp"


namespace np5 {

	/** @brief Functor to compute L1 error
	 */
	template <typename It>
	class l1_estimation {
	public:
		l1_estimation(It first, It last) noexcept
			: first_(first), last_(last) {}

		double operator()(std::valarray<double> const& p) const noexcept {
			double r = 0;
			for (It iter = first_; iter != last_; ++iter)
				r += std::fabs(mcore::eval(p, iter->x) - iter->y);
			return r;
		}

	private:
		It const first_;
		It const last_;
	};

	/** @brief Given a data set approximates it with a polynom
	 */
	template <typename It>
	std::valarray<double> approximate_l1(It begin, It end, size_t const degree) {
		std::valarray<double> p(0.7, degree + 1);
		p[0] = 0.1;
		p[1] = 0.2;
		p[2] = 0.03;
		//p[3] = 0.4;
		mcore::conf_opt conf;
		conf.initial_step = 0.1;
		conf.num_iterations = 51000;
		l1_estimation<It> F(begin, end);
		return mcore::optimize_hj(F, p, conf);
	}

}
