# pragma once

# include <cassert>
# include <memory>
# include <cstring>
# include <type_traits>

namespace mcore { namespace calc {

	struct configuration_NM {
		double reflection_coeff() noexcept { return 1; }
		double expansion_coeff() noexcept { return 2; }
		double contraction_coeff() noexcept { return 0.5; }
		double reduction_coeff() noexcept { return 0.5; }

		size_t num_iterations() const noexcept { return 3000; }
	};
	
	/** @brief Copy one vector to another
	 *
	 * This operation helps us save few instructions
	 *
	 */
	template <typename P>
	void copy(P& to, P const& from) {
		assert(to.size() == from.size());
		for (size_t i = 0; i < to.size(); ++i)
			to[i] = from[i];
	}


	template <typename P>
	class simplex {
		struct extended_point {
			P point{};
			double value{0};
		};

	public:
		simplex() noexcept
			: points2_(nullptr) {}

		template <typename F>
		simplex(F&& func, P const& pt, double delta) : points2_(nullptr), centroid_(pt.size()) {
			reset(func, pt, delta);
		}
		
		simplex(simplex&) = delete;
		simplex& operator=(simplex&) = delete;

		/** @brief Returns dimension of the simplex
		 */
		size_t size() const noexcept {
			return (points2_ != nullptr) 
				? points2_[0].point.size() + 1
				: 0;
		}

		/** @brief Initiates simplex
		 *
		 * @param func    function to be minimized
		 * @param initial initial point of the algorithm
		 * @param delta
		 */
		template <typename F>
		void reset(F const& func, P const& initial, double delta) {
			size_t const dim = initial.size();

			// Allocate memory for the simplex
			points2_.reset(new extended_point[dim + 1]);

			// Compute simplex
			points2_[0].point = std::move(initial.copy());
			points2_[0].value = func(initial);

			for (size_t i = 1; i < dim + 1; ++i) {
				auto& p = points2_[i];
				p.point = std::move(initial.copy());
				p.point[i - 1] *= (1 + delta);
				p.value = func(p.point);
			}
		}

		/** @brief Performs preparation step
		 */
		void update() {
			// Find out the reference points
			buckling_[0] = &points2_[0];
			buckling_[1] = buckling_[0];
			buckling_[2] = buckling_[0];

			size_t const dim = points2_[0].point.size();
			for (size_t i = 0; i < dim + 1; ++i) {
				auto& p = points2_[i];
				if (p.value < buckling_[0]->value)
					buckling_[0] = &p;
				else {
					if (p.value > buckling_[2]->value) {
						buckling_[1] = buckling_[2];
						buckling_[2] = &p;
					}

					if (buckling_[1]->value < p.value && p.value < buckling_[2]->value)
						buckling_[1] = &p;
				}
			}

			// Compute centroid
			for (size_t i = 0; i < dim; ++i)
				centroid_[i] = 0;
			for (size_t i = 0; i < dim + 1; ++i) {
				auto const& p = points2_[i];
				if (&p != buckling_[2])
					centroid_ += p.point;
			}
			centroid_ /= double(dim);
		}

		/** @brief Returns point dispersion
		 */
		double get_point_spreading() const noexcept {
			size_t const dim = points2_[0].point.size();
			double r = 0;
			for (size_t i = 0; i < dim + 1; ++i) {
				double l = 0;
				P& p = points2_[i].point;
				for (size_t j = 0; j < dim; ++j) {
					double const dx = p[j] - centroid_[j];
					l += dx * dx;
				}
				r += l;
			}
			return r / (dim + 1);
		}


		P const& centroid() const noexcept {
			return centroid_;
		}

		template <size_t N>
		extended_point& get_reference() noexcept {
			return *buckling_[N];
		}

		/** @brief Returns vertex of the simplex
		 *
		 * Please pay attention that operation is not safe.
		 * When index >= size() it is expected UB.
		 */
		extended_point& operator[](size_t index) noexcept {
			assert(index < size());
			return points2_[index];
		}

	private:
		std::unique_ptr<extended_point[]> points2_;
		extended_point* buckling_[3];
		P centroid_;
	};
	

	template <typename F, typename P, typename C>
	typename std::remove_reference<P>::type 
	optimize_NM(F&& func, P const& initial, C config) {
		typedef typename std::remove_reference<P>::type point_type;
		
		simplex<point_type> smp(func, initial, 0.05);

		double const alpha = config.reflection_coeff();
		double const beta  = config.expansion_coeff();
		double const gamma = config.contraction_coeff();
		double const delta = config.reduction_coeff();
	
		P xr = initial.copy();
		P xe = initial.copy();
		P xc = initial.copy();

		size_t iter = 0;
		while (iter++ < config.num_iterations()) {
			// Step 1: "sorting" and centroid comptation
			smp.update();

			if (smp.get_point_spreading() < 1.e-12) 
				break;

			auto const& x0 = smp.centroid();
			auto& p2 = smp.template get_reference<2>();
			auto& p1 = smp.template get_reference<1>();
			auto& p0 = smp.template get_reference<0>();

			// Reflection step
			xr = x0 + alpha * (x0 - p2.point);
			double const fr = func(xr);

			if ((p0.value <= fr) && (fr <= p1.value)) {
				copy(p2.point, xr);
				p2.value = fr;
				continue;
			}


			// Expansion step
			if (fr < p0.value) {
				xe = x0 + beta * (xr - x0);
				double const fe = func(xe);
				if (fe < fr) {
					copy(p2.point, xe);
					p2.value = fe;
				} else {
					copy(p2.point, xr);
					p2.value = fr;
				}
				continue;
			}

			// Contraction step
			xc = (fr < p2.value) 
				? x0 + gamma * (xr - x0)
				: x0 + gamma * (p2.point - x0);
			double const fc = func(xc);
			if (fc < fr) {
				copy(p2.point, xc);
				p2.value = fc;
				continue;
			}

			// Reduction step
			for (size_t i = 0; i < smp.size(); ++i) {
				auto& pt = smp[i];
				if (&pt != &p0) {
					pt.point = pt.point + delta * (pt.point - p0.point);
					pt.value = func(pt.point);
				}
			}
		}

		return smp.centroid().copy();
	}

}} // namespace calc / mcore
