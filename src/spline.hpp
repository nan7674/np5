# pragma once

# include <algorithm>
# include <vector>
# include <memory>
# include <functional>

# include "common.hpp"
# include "mcore.hpp"

# include <iostream>


namespace np5_1 {

	static const size_t SPLINE_DEGREE = 3;

	struct spline_node {
		double x;
		double w[SPLINE_DEGREE + 1];

		spline_node(double xx, double a, double b, double c, double d) noexcept 
				: x(xx) {
			w[0] = a;
			w[1] = b;
			w[2] = c;
			w[3] = d;
		}

		spline_node(
				double const x0, double const x1,
				double const y0, double const y1,
				double const m0, double const m1) noexcept : x(x0) {
			double const h = x1 - x0;
			w[2] = m0;
			w[3] = (m1 - m0) / (3. * h);
			w[0] = y0;
			w[1] = (y1 - y0) / h - h * (w[2] + w[3] * h);
		}


		double operator()(double v) const noexcept {
			double const t = v - x;
			return w[0] + t * (w[1] + t * (w[2] + w[3] * t));
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
		typedef std::vector<np5_1::spline_node> node_container;

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

	private:
		std::vector<spline_node> nodes_;
	};


	template <typename V>
	class spl_mem_alloc {
		typedef V value_type;

	public:
		spl_mem_alloc() noexcept 
			: buffer_size_(0) {}

		void allocate(size_t const num_nodes, value_type*& D0, value_type*& D1, value_type*& D2, value_type*& b) {
			size_t const required_size = get_approx_memory(num_nodes);
			reallocate(required_size);
			set_pointers(num_nodes, D0, D1, b);
			D2 = b + num_nodes;
		}

		void allocate(size_t const num_nodes, value_type*& D0, value_type*& D1, value_type*& b) {
			size_t const required_size = get_interp_memory(num_nodes);
			reallocate(required_size);
			set_pointers(num_nodes, D0, D1, b);
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
			return 3 * num_nodes - 4;
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

		void set_pointers(size_t const num_nodes, value_type*& D0, value_type*& D1, value_type*& b) noexcept {
			D0 = buffer_.get();
			D1 = D0 + num_nodes - 2;
			b = D1 + num_nodes - 2;
		}

	private:
		std::unique_ptr<value_type> buffer_;
		size_t buffer_size_;
	};


	class spline_boundary {
	public:
		spline_boundary() {}

		/*! @brief Fills boundary condition for splines
		 */
		virtual void fill_fringe(double* const bs, size_t const size) const;
	};

	class d2_boundary : public spline_boundary {
	public:
		d2_boundary(double const m0, double const m1) noexcept
			: m0_(0), m1_(m1) {}

		/*! @brief Fills boundary condition for splines
		*/
		virtual void fill_fringe(double* const bs, size_t const size) const;

	private:
		double const m0_;
		double const m1_;
	};


	/*! @brief Class for spline creation
	 */
	class spline_builder {
		typedef double value_type;

	public:
		spline_builder() {}

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

			for (size_t i = 0; i < num_nodes - 2; ++i)
				std::cout << i << " : " << D0[i] << ' ' << D1[i] << ' ' << D2[i] << ' ' << b[i + 1] << '\n';

			mcore::cholesky(D0, D1, D2, num_nodes - 2);
			mcore::solve_ldl(D0, D1, D2, b + 1, num_nodes - 2);


			for (size_t i = 0; i < num_nodes - 2; ++i)
				std::cout << i << " : " << ' ' << b[i + 1] << '\n';

			return compute_coefficients(iter, num_nodes, b + 1, Q);
		}

		/** @brief Creates optimal spline
		 */
		template <typename It>
		spline build_optimal(It b, It e) {}

		template <typename It>
		spline operator()(It iter, It itere, spline_boundary const& fringe = spline_boundary()) {
			size_t const num_nodes = get_num_nodes(iter, itere);
			value_type *D0 = nullptr, *D1 = nullptr, *b = nullptr;
			alloc_.allocate(num_nodes, D0, D1, b);

			init_matrices(iter, num_nodes, D0, D1, b + 1);
			fringe.fill_fringe(b, num_nodes);

			mcore::cholesky(D0, D1, num_nodes - 2);
			mcore::solve_ldl(D0, D1, b + 1, num_nodes - 2);

			return compute_coefficients(iter, num_nodes, b + 1);
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


				D0[no] = Q * (mcore::sqr(r0) + mcore::sqr(r1) + mcore::sqr(r2)) + 2. * (h0 + h1);
				D1[no] = Q * r2 * (r1 + r3) + h1;
				D2[no] = 3. * Q * r2 / h2;

				b[no] = r0 * y0 + r1 * y1 + r2 * y2;

				x0 = x1, y0 = y1;
				x1 = x2, y1 = y2;
			}
		}

		template <typename It>
		static void init_matrices(
				It iter, size_t num_nodes,
				value_type* const D0,
				value_type* const D1,
				value_type* const b) {

			value_type x0 = iter->x, y0 = iter->y;
			++iter;
			value_type x1 = iter->x, y1 = iter->y;
			++iter;

			for (size_t no = 0; no < num_nodes - 2; ++iter, ++no) {
				double const x2 = iter->x, y2 = iter->y;

				double const h0 = x1 - x0;
				double const h1 = x2 - x1;

				double const r0 = 3. / h0;
				double const r2 = 3. / h1;
				double const r1 = -(r0 + r2);

				D0[no] = 2. * (h0 + h1);
				D1[no] = h1;

				b[no] = r0 * y0 + r1 * y1 + r2 * y2;

				x0 = x1, y0 = y1;
				x1 = x2, y1 = y2;
			}
		}

		template <typename It>
		static std::vector<spline_node> compute_coefficients(
				It iter, size_t const num_nodes,
				double const* const b) {

			std::vector<spline_node> S;
			S.reserve(num_nodes - 1);

			It p0 = iter++;
			It p1 = iter++;

			{
				It p2 = iter++;
				S.emplace_back(spline_node(p0->x, p1->x, p0->y, p1->y, *(b - 1), b[0]));

				p0 = p1;
				p1 = p2;
			}

			for (size_t i = 2; i < num_nodes - 2; ++i, ++iter) {
				It p2 = iter;

				S.emplace_back(spline_node(p0->x, p1->x, p0->y, p1->y, b[i - 2], b[i - 1]));

				p0 = p1;
				p1 = p2;
			}


			// Compute the very last coefficients.
			{
				S.push_back(spline_node(p0->x, p1->x, p0->y, p1->y, b[num_nodes - 3], b[num_nodes - 2]));
				It p2 = p1 + 1;
				S.push_back(spline_node(p1->x, p2->x, p1->y, p2->y, b[num_nodes - 2], b[num_nodes - 1]));
			}

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

				y0 = p1->y - Q * (r1 * b[1] + (r0 + r1) * b[0]);
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
	};

}
