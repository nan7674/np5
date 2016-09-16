# include "utils/data.hpp"

# include <random>
# include <limits>
# include <chrono>

namespace {

class environment {
public:
	double tolerance() const noexcept {
		return tolerance_;
	}

	double& tolerance() noexcept {
		return tolerance_;
	}

private:
	double tolerance_{1.e-16};
};

typedef np5::singleton<environment> env;

} // anonymous namespace


namespace tmr = std::chrono;


// Creates random poly of a given degree
mcore::calc::polynom np5::utils::create_random_polynom(size_t degree) {
	mcore::calc::polynom p(degree);

	std::mt19937 generator(std::random_device{}());
	std::uniform_real_distribution<> D(-1, 1);

	for (size_t i = 0; i < degree + 1; ++i)
		p[i] = D(generator);

	return p;
}

void np5::utils::add_outliers(std::vector<np5::point>& data, size_t nouts) {
	if (data.empty())
		return;

	double ymin = data.front().y;
	double ymax = ymin;

	for (auto iter = std::next(data.begin()); iter != data.end(); ++iter) {
		double const v = iter->y;
		if (v > ymax)
			ymax = v;
		else if (v < ymin)
			ymin = v;
	}

	std::mt19937 generator(std::random_device{}());

	std::uniform_real_distribution<> D0(0, 1);
	std::uniform_real_distribution<> D1(ymin, ymax);
	for (int i = data.size() - 1; i > -1; --i) {
		double const thr = nouts / (i + 1);
		double const p = D0(generator);
		if (thr < p) {
			data[i].y = D1(generator);
			--nouts;
		}
	}
}

bool np5::utils::approx::operator==(double const other) const noexcept {
	double const denom = std::abs(value_) + std::abs(other) +
		std::numeric_limits<double>::min();
	double const diff = 2 * std::abs(value_ - other);
	double const tol = std::isnan(tolerance_)
		? env::instance().tolerance()
		: tolerance_;
	return (diff / denom) < tol;
}

void np5::utils::set_default_tolerance(double tol) noexcept {
	env::instance().tolerance() = tol;
}

/*! @brief Given the function runs it and measure time of execution
*
* @param[in] f function to run
* @return elapsed time in seconds
*/
double np5::utils::measure_time(std::function<void()> f) {
	auto const start = tmr::system_clock::now();
	f();
	auto const duration = tmr::duration_cast<tmr::milliseconds>(
		tmr::system_clock::now() - start);
	return duration.count() / 1000.;
}
