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


/* @brief Cholesky decomposition of a symmetric matrices
 *
 * @param d0 a main diagonal of the matrix
 */
void mcore::cholesky(
		double* const d0,
		double* const d1,
		double* const d2,
		size_t const n) noexcept {

	d1[0] /= d0[0];
	d2[0] /= d0[0];

	d0[1] -= sqr(d1[0]) * d0[0];
	d1[1] = (d1[1] - d0[0] * d1[0] * d2[0]) / d0[1];
	d2[1] /= d0[1];

	for (size_t i = 2; i < n - 2; ++i) {
		d0[i] -= d0[i - 2] * sqr(d2[i - 2]) + d0[i - 1] * sqr(d1[i - 1]);
		d1[i] = (d1[i] - d0[i - 1] * d1[i - 1] * d2[i - 1]) / d0[i];
		d2[i] /= d0[i];
	}

	d0[n - 2] -= d0[n - 4] * sqr(d2[n - 4]) + d0[n - 3] * sqr(d1[n - 3]);
	d1[n - 2] = (d1[n - 2] - d0[n - 3] * d1[n - 3] * d2[n - 3]) / d0[n - 2];

	d0[n - 1] -= d0[n - 3] * sqr(d2[n - 3]) + d0[n - 2] * sqr(d1[n - 2]);
}


/* @brief Cholesky decomposition of a symmetric matrices
*
* @param d0 a main diagonal of the matrix
*/
void mcore::cholesky(
		double* const d0,
		double* const d1,
		size_t const n) noexcept{

	d1[0] /= d0[0];

	d0[1] -= sqr(d1[0]) * d0[0];
	d1[1] /= d0[1];

	for (size_t i = 2; i < n - 2; ++i) {
		d0[i] -= d0[i - 1] * sqr(d1[i - 1]);
		d1[i] /= d0[i];
	}

	d0[n - 2] -= d0[n - 3] * sqr(d1[n - 3]);
	d1[n - 2] /= d0[n - 2];

	d0[n - 1] -= d0[n - 2] * sqr(d1[n - 2]);
}


void mcore::solve_ldl(
		double const* const d0,
		double const* const d1,
		double const* const d2,
		double* const y,
		size_t const n) noexcept {
	y[1] -= d1[0] * y[0];

	for (size_t i = 2; i < n; ++i)
		y[i] -= d1[i - 1] * y[i - 1] + d2[i - 2] * y[i - 2];

	for (size_t i = 0; i < n; ++i)
		y[i] /= d0[i];

	y[n - 2] -= d1[n - 2] * y[n - 1];

	for (size_t i = n - 3; i < std::numeric_limits<size_t>::max(); --i)
		y[i] -= d1[i] * y[i + 1] + d2[i] * y[i + 2];
}


void mcore::solve_ldl(
		double const* const d0,
		double const* const d1,
		double* const y,
		size_t const n) noexcept{
	y[1] -= d1[0] * y[0];

	for (size_t i = 2; i < n; ++i)
		y[i] -= d1[i - 1] * y[i - 1];

	for (size_t i = 0; i < n; ++i)
		y[i] /= d0[i];

	y[n - 2] -= d1[n - 2] * y[n - 1];

	for (size_t i = n - 3; i < std::numeric_limits<size_t>::max(); --i)
		y[i] -= d1[i] * y[i + 1];
}


mcore::poly_type mcore::get_random_poly(
		size_t const degree, double const cmin, double const cmax) {
	poly_type poly(degree + 1);
	RG& g = RG::instance();
	for (size_t i = 0; i < degree + 1; ++i)
		poly[i] = g.double_uniform(cmin, cmax);
	return poly;
}


mcore::mat mcore::mat::identity(size_t const dim, double const value) {
	mat retval(dim, dim);
	for (size_t i = 0; i < dim; ++i)
		retval(i, i) = value;
	return retval;
}

namespace {

	typedef double numeric_type;

	double const EPS = 1.22e-16;



	/*
 * This function returns the solution of Ax = b
 *
 * The function employs LU decomposition followed by forward/back substitution (see
 * also the LAPACK-based LU solver above)
 *
 * A is mxm, b is mx1
 *
 * The function returns 0 in case of error, 1 if successful
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
int AX_EQ_B_LU(numeric_type* A, numeric_type* B, numeric_type* x, size_t m)
{
	void*  buf = nullptr;
	size_t buf_sz = 0;

	int j, k;
	int *idx, maxi=-1, idx_sz, a_sz, work_sz;
	double *a, *work, tmp;

//    if(!A)
//#ifdef LINSOLVERS_RETAIN_MEMORY
//    {
//      if(buf) free(buf);
//      buf=NULL;
//      buf_sz=0;

//      return 1;
//    }
//#else
//    return 1; /* NOP */
//#endif /* LINSOLVERS_RETAIN_MEMORY */

  /* calculate required memory size */
  idx_sz = m;
  a_sz = m * m;
  work_sz = m;
	size_t const total_size = (a_sz + work_sz) * sizeof(double) + idx_sz * sizeof(int); /* should be arranged in that order for proper doubles alignment */

#ifdef LINSOLVERS_RETAIN_MEMORY
  if(tot_sz>buf_sz){ /* insufficient memory, allocate a "big" memory chunk at once */
    if(buf) free(buf); /* free previously allocated memory */

    buf_sz=tot_sz;
    buf=(void *)malloc(tot_sz);
    //if(!buf){
    //  fprintf(stderr, RCAT("memory allocation in ", AX_EQ_B_LU) "() failed!\n");
    //  exit(1);
    //}
  }
#else
    buf_sz = total_size;
    buf=(void *)malloc(total_size);
    //if(!buf){
    //  fprintf(stderr, RCAT("memory allocation in ", AX_EQ_B_LU) "() failed!\n");
    //  exit(1);
    //}
#endif /* LINSOLVERS_RETAIN_MEMORY */

	a = reinterpret_cast<double*>(buf);
	work = a + a_sz;
	idx = (int*)(work + work_sz);

  /* avoid destroying A, B by copying them to a, x resp. */
	memcpy(a, A, a_sz * sizeof(numeric_type));
	memcpy(x, B, m * sizeof(numeric_type));

  /* compute the LU decomposition of a row permutation of matrix a; the permutation itself is saved in idx[] */
	for (size_t i = 0; i < m; ++i) {
		double max_value = 0;
		for(j=0; j<m; ++j)
			if((tmp = std::abs(a[i*m+j])) > max_value)
				max_value = tmp;
//		  if(max==0.0){
//        fprintf(stderr, RCAT("Singular matrix A in ", AX_EQ_B_LU) "()!\n");
//#ifndef LINSOLVERS_RETAIN_MEMORY
//        free(buf);
//#endif

//        return 0;
//      }
		  work[i] = 1. / max_value; //LM_CNST(1.0)/max;
	}

	for (j = 0; j < m; ++j) {
		for (size_t i = 0; i < j; ++i) {
			numeric_type sum = a[i * m + j];
			for(k =0 ; k < i; ++k)
				sum -= a[i * m + k] * a[k * m + j];
			a[i * m + j] = sum;
		}

		double max_value = 0;
		for(size_t i = j; i < m; ++i) {
			numeric_type sum = a[i * m + j];
			for(k=0; k<j; ++k)
				sum -= a[i * m + k] * a[k * m + j];
			a[i * m + j] = sum;
			if ((tmp=work[i] * std::abs(sum))>=max_value) {
				max_value = tmp;
				maxi=i;
			}
		}
		if(j!=maxi){
			for(k=0; k<m; ++k){
				tmp=a[maxi*m+k];
				a[maxi*m+k]=a[j*m+k];
				a[j*m+k]=tmp;
			}
			work[maxi]=work[j];
		}
		idx[j]=maxi;
		if(a[j*m+j]==0.0)
      a[j*m+j]=EPS; //LM_REAL_EPSILON;
		if(j!=m-1){
			tmp = 1./ a[j * m + j]; //LM_CNST(1.0)/(a[j*m+j]);
			for(size_t i = j + 1; i<m; ++i)
				a[i*m+j]*=tmp;
		}
	}

  /* The decomposition has now replaced a. Solve the linear system using
   * forward and back substitution
   */
	for (size_t i=k=0; i<m; ++i) {
		j = idx[i];
		numeric_type sum = x[j];
		x[j]=x[i];
		if (k!=0)
			for(j=k-1; j<i; ++j) sum-=a[i*m+j]*x[j];
		else if(sum!=0.0)
			k=i+1;
		x[i]=sum;
	}

	for(size_t i = m - 1; i < m; --i) {
		numeric_type sum=x[i];
		for(j=i+1; j<m; ++j)
      sum-=a[i*m+j]*x[j];
		x[i]=sum/a[i*m+i];
	}

#ifndef LINSOLVERS_RETAIN_MEMORY
  free(buf);
#endif

  return 1;
}


}

/** @brief Solves linear system A x= b.
	*
	* @param A
	*/
mcore::vec mcore::solve(mat const& A, vec const& b) {
	// TODO :: remove data copying
	std::vector<double> A1(9), b1(3), x(3);
	for (size_t i = 0; i < 3; ++i)
		b1[i] = i;

	for (size_t i = 1; i < 10; ++i)
		A1[i - 1] = i * i;

	AX_EQ_B_LU(A1.data(), b1.data(), x.data(), 3);

	for (size_t i = 0; i < 3; ++i)
		std::cout << x[i] << std::endl;
	return vec();
}

