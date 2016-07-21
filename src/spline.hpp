# pragma once

# include <algorithm>
# include <functional>
# include <vector>
# include <memory>
# include <stdexcept>

# include "common.hpp"

# include "mcore/linalg.hpp"
# include "mcore/calc.hpp"


namespace np5 {

static const size_t SPLINE_DEGREE = 3;

struct spline_node {
	double x;
	double w[SPLINE_DEGREE + 1];

	spline_node(double xx, double a, double b, double c, double d) noexcept
		: x(xx)
	{
		w[0] = a;
		w[1] = b;
		w[2] = c;
		w[3] = d;
	}

	spline_node(
		double const x0, double const x1,
		double const y0, double const y1,
		double const m0, double const m1) noexcept
		: x(x0)
	{
		double const h = x1 - x0;
			w[2] = m0;
			w[3] = (m1 - m0) / (3. * h);
			w[0] = y0;
			w[1] = (y1 - y0) / h - h * (w[2] + w[3] * h);
	}

	double operator()(double v) const noexcept {
		double const t = v - x;
		return w[0] + t * (w[1] + t * (w[2] + t * w[3]));
	}

	std::tuple<double, double, double, double> get_global() const noexcept {
		double const x2 = x * x;
		double const x3 = x2 * x;

		double const w0 = w[0] - w[1] * x + w[2] * x2 - w[3] * x3;
		double const w1 = w[1] - 2 * w[2] * x + 3 * w[3] * x2;
		double const w2 = w[2] - 3 * w[3] * x;

		return std::make_tuple(w0, w1, w2, w[3]);
	}
};


class spline {
	typedef std::vector<spline_node> node_container;

public:
	spline() noexcept {}

	spline(spline&& other) noexcept
		: nodes_(std::move(other.nodes_)) {}

	spline(node_container&& ns) noexcept
		: nodes_(std::move(ns)) {}

	spline& operator=(spline&& other) noexcept {
		if (this != &other) {
			nodes_ = std::move(other.nodes_);
		}

		return *this;
	}

	double operator()(double const x) const noexcept {
		if (x <= nodes_.front().x)
			return nodes_.front()(x);
		else {
			auto iter = std::lower_bound(std::begin(nodes_), std::end(nodes_), x,
				[](decltype(nodes_.front())& t, double const x) { return t.x < x; }) - 1;

			return (*iter)(x);
		}
	}

	node_container const& nodes() const noexcept {
		return nodes_;
	}

private:
	node_container nodes_;
};


template <typename V>
class spl_mem_alloc {
	typedef V value_type;

public:
	spl_mem_alloc() noexcept
		: buffer_size_(0) {}

	//void allocate(size_t const num_nodes, value_type*& D0, value_type*& D1, value_type*& D2, value_type*& b)
	//{
	//	size_t const required_size = get_approx_memory(num_nodes);
	//	reallocate(required_size);
	//	set_pointers(num_nodes, D0, D1, b);
	//	D2 = b + num_nodes;
	//}

	void allocate(
		size_t const num_nodes,
		value_type*& D0,
		value_type*& D1,
		value_type*& D2,
		value_type*& b)
	{
		size_t const required_size = get_interp_memory(num_nodes);
		reallocate(required_size);
		set_pointers(num_nodes, D0, D1, D2, b);
	}

private:
	spl_mem_alloc(spl_mem_alloc const&);
	spl_mem_alloc& operator=(spl_mem_alloc const&);

	spl_mem_alloc(spl_mem_alloc&&);
	spl_mem_alloc& operator=(spl_mem_alloc&&);

private:
	static size_t get_approx_memory(size_t const num_nodes) noexcept {
		return 4 * num_nodes - 6;
	}

	static size_t get_interp_memory(size_t const num_nodes) noexcept {
		return 4 * num_nodes;
	}

	void reallocate(size_t const required_size) {
		if (required_size > buffer_size_) {
			buffer_.reset(new V[required_size]);
			buffer_size_ = required_size;
# ifdef _DEBUG
			memset(buffer_.get(), 0, buffer_size_ * sizeof(value_type));
# endif
		}
	}

	/*! \@brief Allocates a memory for a spline construction
	 */
	void set_pointers(
		size_t const num_nodes,
		value_type*& D0,
		value_type*& D1,
		value_type*& D2,
		value_type*& b)
	{
		D0 = buffer_.get();
		D1 = D0 + num_nodes;
		D2 = D1 + num_nodes;
		b = D2 + num_nodes;
	}

private:
	std::unique_ptr<value_type> buffer_;
	size_t buffer_size_;
};


struct spline_fringe {
	double f0;
	double f1;
	double h0;

	double g0;
	double g1;
	double hn;
};


class spline_boundary {
public:
	spline_boundary() {}

	/*! \brief Fills boundary condition for splines
	 */
	virtual void fill_fringe(
		double* diag,
		double* bs,
		spline_fringe const& boundary,
		size_t size) const noexcept;

	/*! \brief Update spline coefficients
	 */
	virtual void update_coefficients(
		double* cs,
		spline_fringe const& boundary,
		size_t size) const noexcept;
};


class d2_boundary : public spline_boundary {
public:
	d2_boundary(double const m0, double const m1) noexcept
		: m0_(0.5 * m0), m1_(0.5 * m1) {}

	/*! @brief Fills boundary condition for splines
	 */
	void fill_fringe(
		double* diag,
		double* bs,
		spline_fringe const& boundary,
		size_t const size) const noexcept override;

	/*! \brief Update spline coefficients
	 */
	void update_coefficients(
		double* cs,
		spline_fringe const& boundary,
		size_t size) const noexcept override;

private:
	double m0_;
	double m1_;
};


class d1_boundary : public spline_boundary {
public:
	d1_boundary(double const v0, double const v1) noexcept
		: fp0_(v0), fp1_(v1) {}

	/*! @brief Fills boundary condition for splines
	 */
	void fill_fringe(
		double* diag,
		double* bs,
		spline_fringe const& boundary,
		size_t const size) const noexcept override;

	/*! \brief Update spline coefficients
	 */
	void update_coefficients(
		double* cs,
		spline_fringe const& boundary,
		size_t size) const noexcept override;

private:
	double get_k0(spline_fringe const& boundary) const noexcept;
	double get_kn(spline_fringe const& boundary) const noexcept;

private:
	double fp0_;
	double fp1_;
};


/*! @brief Class for spline creation
 */
class spline_builder {
	typedef double value_type;

public:
	spline_builder() = default;

	/*! @brief Creates smoothing spline
	 */
	template <typename It>
	spline operator()(It iter, It itere, double lambda, spline_boundary const& fr = spline_boundary()) {
		size_t const num_nodes = get_num_nodes(iter, itere);
		value_type *D0 = nullptr, *D1 = nullptr, *D2 = nullptr, *b = nullptr;
		alloc_.allocate(num_nodes, D0, D1, D2, b);

		double const Q = 2. / 3. * lambda / (1 - lambda);

		b[0] = 0;
		b[num_nodes - 1] = 0;

		init_matrices(iter, num_nodes, D0, D1, D2, b + 1, Q);

		// TODO :: need to add a solver for a five-band matrix
		//::mcore::linalg::cholesky(D0, D1, D2, num_nodes - 2);
		//::mcore::linalg::solve_ldl(D0, D1, D2, b + 1, num_nodes - 2);

		return compute_coefficients(iter, num_nodes, b + 1, Q);
	}

	/** @brief Computes an interpolational spline.
	 */
	template <typename It>
	spline operator()(
		It iter, It itere,
		spline_boundary const& fringe = spline_boundary())
	{
		size_t const num_nodes = get_num_nodes(iter, itere);
		if (num_nodes == 0)
			throw std::runtime_error("empty data");

		value_type *D0(nullptr), *D1(nullptr), *D2(nullptr), *b(nullptr);
		alloc_.allocate(num_nodes, D0, D1, D2, b);

		init_matrices(iter, num_nodes, D0 + 1, D1, D2, b + 1);
		fringe.fill_fringe(D1, b, fringe_, num_nodes);

		mcore::linalg::solve_tridiagonal(D0, D1, D2, b + 1, num_nodes - 2);
		fringe.update_coefficients(b, fringe_, num_nodes);

		return compute_coefficients(iter, num_nodes, b);
	}

private:
	template <typename It>
	static size_t get_num_nodes(It iter, It itere) {
		size_t const np = std::distance(iter, itere);
		return np;
	}

	template <typename It>
	static void init_matrices(
			It iter, size_t num_nodes,
			value_type* const D0,
			value_type* const D1,
			value_type* const D2,
			value_type* const b,
			double const Q) {

		value_type x0 = iter->x, y0 = iter->y;
		++iter;
		value_type x1 = iter->x, y1 = iter->y;
		++iter;

		double const H0 = x1 - x0;

		for (size_t no = 0; no < num_nodes - 2; ++iter, ++no) {
			double const x2 = iter->x, y2 = iter->y;
			double const x3 = no + 3 < num_nodes ? (iter + 1)->x : 0;

			double const h0 = x1 - x0;
			double const h1 = x2 - x1;
			double const h2 = x3 - x2;

			double const r0 = 3. / h0;
			double const r2 = 3. / h1;
			double const r1 = -(r0 + r2);
			double const r3 = -r2 - 3. / h2;


			D0[no] = Q * (mcore::calc::sqr(r0) + mcore::calc::sqr(r1) + mcore::calc::sqr(r2)) + 2. * (h0 + h1);
			D1[no] = Q * r2 * (r1 + r3) + h1;
			D2[no] = 3. * Q * r2 / h2;

			b[no] = r0 * y0 + r1 * y1 + r2 * y2;

			x0 = x1, y0 = y1;
			x1 = x2, y1 = y2;
		}
	}

	template <typename It>
	void init_matrices(
		It iter, size_t num_nodes,
		value_type* const __restrict__ D0,
		value_type* const __restrict__ D1,
		value_type* const __restrict__ D2,
		value_type* const __restrict__ b)
	{
		value_type x0 = iter->x, y0 = iter->y;
		++iter;
		value_type x1 = iter->x, y1 = iter->y;
		++iter;

		fringe_.f0 = y0;
		fringe_.f1 = y1;
		fringe_.h0 = x1 - x0;

		double h0, h1;

		for (size_t no = 0; no < num_nodes - 2; ++iter, ++no) {
			double const x2 = iter->x, y2 = iter->y;

			h0 = x1 - x0;
			h1 = x2 - x1;

			if (h0 <= 0 || h1 <= 0)
				throw std::runtime_error("");

			double const r0 = 3. / h0;
			double const r2 = 3. / h1;
			double const r1 = -(r0 + r2);

			D1[no] = 2. * (h0 + h1);
			D2[no] = D0[no] = h1;

			b[no] = r0 * y0 + r1 * y1 + r2 * y2;

			x0 = x1, y0 = y1;
			x1 = x2, y1 = y2;
		}

		fringe_.g0 = y0;
		fringe_.g1 = y1;
		fringe_.hn = x1 - x0;
	}

	/** @brief Computes coefficients of the spline
	 *
	 * @param iter0     iterator pointing to the first point in the range
	 * @param num_nodes total number of points
	 * @param b         values of an function to interpolate
	 *
	 * The routine is being used for construction of an
	 * interpolation spline.
	 */
	template <typename It>
	static std::vector<spline_node> compute_coefficients(
			It iter0, size_t const num_nodes,
			double const* const b) {

		std::vector<spline_node> S;
		S.reserve(num_nodes - 1);

		It iter1 = iter0;
		++iter1;

		double const* w0 = b;
		double const* w1 = b + 1;
		for (size_t i = 0; i < num_nodes - 1; ++i, ++iter0, ++iter1, ++w0, ++w1)
			S.emplace_back(spline_node(
				iter0->x, iter1->x,
				iter0->y, iter1->y,
				*w0, *w1));

		return std::move(S);
	}

	template <typename It>
	static std::vector<spline_node> compute_coefficients(
		It iter, size_t const num_nodes,
		double const* const b,
		double const Q) {

		std::vector<spline_node> S;
		S.reserve(num_nodes - 1);

		double y0 = 0;

		It p0 = iter++;
		It p1 = iter++;

		{
			It p2 = iter++;

			double const r0 = 3. / (p1->x - p0->x);
			double const r1 = 3. / (p2->x - p1->x);

			double const y = p0->y - Q * r0 * b[0];

			y0 = p1->y - Q * (r1 * b[0] + (r0 + r1) * b[1]);
			S.emplace_back(spline_node(p0->x, p1->x, y, y0, *(b - 1), b[0]));

			p0 = p1;
			p1 = p2;
		}

		for (size_t i = 2; i < num_nodes - 2; ++i, ++iter) {
			It p2 = iter;

			double const r1 = 3. / (p1->x - p0->x);
			double const r2 = 3. / (p2->x - p1->x);

			double const y1 = p1->y - Q * (r1 * b[i - 2] - (r1 + r2) * b[i - 1] + r2 * b[i]);
			S.emplace_back(spline_node(p0->x, p1->x, y0, y1, b[i - 2], b[i - 1]));
			y0 = y1;
			p0 = p1;
			p1 = p2;
		}


		// Compute the very last coefficients.
		{
			It p2 = p1 + 1;

			double const r1 = 3. / (p1->x - p0->x);
			double const r2 = 3. / (p2->x - p1->x);

			double y1 = p1->y - Q * (r1 * b[num_nodes - 4] + (r1 + r2) * b[num_nodes - 3]);
			S.push_back(spline_node(p0->x, p1->x, y0, y1, b[num_nodes - 3], b[num_nodes - 2]));
			y0 = y1;

			y1 = p2->y - Q * r2 * b[num_nodes - 3];
			S.push_back(spline_node(p1->x, p2->x, y0, y1, b[num_nodes - 2], b[num_nodes - 1]));
		}

		return std::move(S);
	}

private:
	spl_mem_alloc<value_type> alloc_;
	spline_fringe fringe_;
};

}
