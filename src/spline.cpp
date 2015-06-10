# include "spline.hpp"

# include <limits>



/** @brief Fills boundary condition for splines
 */
void np5::spline_boundary::fill_fringe(double* const bs, size_t const size) const {
	bs[0] = 0;
	bs[size - 1] = 0;
}
