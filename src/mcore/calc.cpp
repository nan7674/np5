# include "calc.hpp"


double& mcore::calc::polynom::operator[](size_t index) {
	if (index < coeffs_.size())
		return coeffs_[index];
	else {
		coeffs_.resize(index + 1);
		return coeffs_[index];
	}
}
