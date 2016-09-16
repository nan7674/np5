# pragma once

# define WITH_LAPACK

// =====================================================================
// Some definitions
// =====================================================================
# ifdef WITH_LAPACK
# define GET_ELEMENT(data, nrows, ncols, row, column) \
	data[(column) * (nrows) + (row)]
# else
# define GET_ELEMENT(data, nrows, ncols, row, column) \
	data[(row) * (ncols) + (column)]
# endif

