# include "polynomial.hpp"

# include <algorithm>
# include <cstdlib>


namespace {

	namespace det = np5::detail;

} // anonymous namespace

det::num_generator::num_generator(size_t upper_bound, size_t count)
	: data_(upper_bound + 1), count_(count),
	  cursor_(0), generator_(rd_())
{
	size_t* ptr = data_.data();
	for (size_t i = 0; i < upper_bound + 1; ++i, ++ptr)
		*ptr = i;

	shuffle();
}

det::num_generator::random_sequence det::num_generator::generate() noexcept {
	size_t* sp = nullptr;
	if (cursor_ + count_ > data_.size()) {
		shuffle();
		sp = data_.data();
		cursor_ = 0;
	} else {
		sp = data_.data() + cursor_;
	}
	cursor_ += count_;
	return random_sequence(sp, count_);

}


inline void det::num_generator::shuffle() noexcept {
	size_t* p1 = data_.data();
	size_t* p2 = data_.data() + data_.size(); // / 2;
	std::shuffle(p1, p2, generator_);
}
