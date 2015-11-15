# include "mcore.hpp"

# include <algorithm>
# include <random>
# include <cstring>

# include "common.hpp"

namespace mcore = np5::mcore;

namespace {

	class random_generator {
	public:
		random_generator() {}

		double double_uniform(double const vmin, double const vmax) {
			return std::uniform_real_distribution<>(vmin, vmax)(gen_);
		}

	private:
		std::mt19937 gen_;
	};

	typedef np5::singleton<random_generator> RG;

}


mcore::poly_type mcore::get_random_poly(
		size_t const degree, double const cmin, double const cmax) {
	poly_type poly(degree + 1);
	RG& g = RG::instance();
	for (size_t i = 0; i < degree + 1; ++i)
		poly[i] = g.double_uniform(cmin, cmax);
	return poly;
}
