# include <Python.h>

# include <cassert>
# include <cstddef>
# include <cstring>
# include <cstdarg>
# include <sstream>
# include <vector>
# include <tuple>

# include <iostream>

# include "polynomial.hpp"
# include "kernel.hpp"


namespace {

class wrapper_exception {
public:
	explicit wrapper_exception(char const* ptr=nullptr)
		: message_(ptr) {}

	wrapper_exception(wrapper_exception const&) = default;
	wrapper_exception& operator=(wrapper_exception const&) = delete;

	char const* message() const noexcept {
		return message_;
	}

private:
	char const* message_;
};


typedef Py_ssize_t (*get_size_function)(PyObject*);
typedef PyObject* (*get_item_function)(PyObject*, Py_ssize_t);

struct point {
	typedef point type;

	point() {}

	point(PyObject* a1, PyObject* a2)
		: x(PyFloat_AsDouble(a1)), y(PyFloat_AsDouble(a2)) {}

	point(PyObject* const object) noexcept
		: x(PyFloat_AsDouble(PyTuple_GetItem(object, 0))),
		  y(PyFloat_AsDouble(PyTuple_GetItem(object, 1))) {}

	point& operator=(PyObject* const object) {
		if (object) {
			x = PyFloat_AsDouble(PyTuple_GetItem(object, 0));
			y = PyFloat_AsDouble(PyTuple_GetItem(object, 1));
		}

		return *this;
	}

	void assign(PyObject* const a1, PyObject* const a2) {
		if (a1 && a2) {
			x = PyFloat_AsDouble(a1);
			y = PyFloat_AsDouble(a2);
		}
	}

	static point pack(PyObject* object) {
		return point(object);
	}

	double x{0};
	double y{0};
};


struct double_type {
	typedef double type;

	static double pack(PyObject* object) {
		return PyFloat_AsDouble(object);
	}
};


char const ERROR_EMPTY_DATA[]        = "Container is empty; nothing to approximate";
char const ERROR_NOT_ALIGNED[]       = "Sequences must be aligned";
char const ERROR_INSUFFICIENT_DATA[] = "Number of samples is too small";
char const ERROR_UNKNOWN_ARGS[]      = "Unknown arguments";


std::tuple<get_item_function, size_t> parse_argument(PyObject* obj) {
	get_item_function getter = nullptr;
	size_t size = 0;
	if (PyTuple_CheckExact(obj)) {
		size = PyTuple_Size(obj);
		getter = PyTuple_GetItem;
	} else if (PyList_CheckExact(obj)) {
		size = PyList_Size(obj);
		getter = PyList_GetItem;
	} else {
		PyErr_SetString(PyExc_ValueError,
			"Unsupported type of an argument");
			throw wrapper_exception();
	}
	return std::make_tuple(getter, size);
}


std::vector<point> parse_data(PyObject* arg1, PyObject* arg2) {
	auto par1 = parse_argument(arg1);
	auto par2 = parse_argument(arg2);
	std::vector<point> pts;

	if (std::get<1>(par1) == std::get<1>(par2)) {
		size_t const size = std::get<1>(par1);
		if (size) {
			pts.reserve(size);
			auto const get1 = std::get<0>(par1);
			auto const get2 = std::get<0>(par2);
			for (size_t i = 0; i < size; ++i)
				pts.emplace_back(get1(arg1, i), get2(arg2, i));
		} else
			throw wrapper_exception(ERROR_EMPTY_DATA);
	} else
			throw wrapper_exception(ERROR_NOT_ALIGNED);

	return pts;
}

template <typename DataType>
std::vector<typename DataType::type> parse_data(PyObject* arg) {
	auto par = parse_argument(arg);
	std::vector<typename DataType::type> points;

	size_t const size = std::get<1>(par);
	if (size) {
		points.reserve(size);
		get_item_function get_item = std::get<0>(par);
		for (size_t i = 0; i < size; ++i)
			points.emplace_back(DataType::pack(get_item(arg, i)));
	} else
		throw wrapper_exception(ERROR_EMPTY_DATA);

	return points;
}




class pyobject_iterator_1 {
public:
	explicit pyobject_iterator_1(get_item_function getter, PyObject* object) noexcept
		: container_(object), getter_(getter)
	{
		assert(container_ != nullptr);
		point_ = getter_(container_, 0);
	}

	pyobject_iterator_1(get_item_function getter, PyObject* object, size_t size) noexcept
		: container_(object), cursor_(size)
	{
		assert(container_ != nullptr);
	}

	bool operator!=(pyobject_iterator_1 const& it) const noexcept {
		assert(container_ == it.container_);
		return cursor_ != it.cursor_;
	}

	pyobject_iterator_1& operator++() noexcept {
		++cursor_;
		point_ = getter_(container_, cursor_);
		return *this;
	}

	point const* operator->() const { return &point_; }

private:
	PyObject* const container_;
	size_t cursor_{0};
	point point_;
	get_item_function getter_;
};

	// This type of iterator incorporates an idea of a ZIP iterator
class pyobject_iterator_2 {
public:
	// Ctor for initial position iterator
	pyobject_iterator_2(
			PyObject* first_sequence,
			get_item_function first_getter,
			PyObject* second_sequence,
			get_item_function second_getter) noexcept
		: s1_{first_sequence}, s2_{second_sequence},
		  get1_{first_getter}, get2_{second_getter},
		  cursor_{0}
	{
		assert(s1_ != nullptr);
		assert(s2_ != nullptr);

		point_.assign(get1_(s1_, 0), get2_(s2_, 0));
	}

	// Ctor for end position iterator
	pyobject_iterator_2(
			PyObject* first_sequence,
			PyObject* second_sequence,
			size_t size) noexcept
		: s1_{first_sequence}, s2_{second_sequence}, cursor_(size)
	{
		assert(s1_ != nullptr);
		assert(s2_ != nullptr);
	}

	bool operator!=(pyobject_iterator_2 const& it) const noexcept {
		assert(s1_ == it.s1_);
		assert(s2_ == it.s2_);
		return cursor_ != it.cursor_;
	}

	pyobject_iterator_2& operator++() noexcept {
		++cursor_;
		point_.assign(get1_(s1_, cursor_), get2_(s2_, cursor_));
		return *this;
	}

	point const* operator->() const noexcept { return &point_; }

private:
	PyObject* s1_;
	PyObject* s2_;

	get_item_function get1_{nullptr};
	get_item_function get2_{nullptr};

	size_t cursor_;
	point point_;
};

struct type_property {
	get_size_function get_size{nullptr};
	get_item_function get_item{nullptr};
};


type_property list_property;
type_property tuple_property;


mcore::linalg::vec
build_l2_approximation(type_property const& prop, PyObject* container, size_t degree) {
	if (Py_ssize_t size = prop.get_size(container)) {
		if (size == 1) {
			// insufficient number of samples;
			// let's generate an exception
			PyErr_SetString(PyExc_ValueError,
				"Number of samples must be greater then 1");
			throw wrapper_exception();
		}

		if (static_cast<size_t>(size) < degree) {
			// number of samples must be greater then
			// desired degree of a plynomial approxmation
			std::ostringstream oss;
			oss << "Insuffient number of samples (";
			oss << size;
			oss << "); degree of a polynom equals to " << degree;
			PyErr_SetString(PyExc_ValueError, oss.str().c_str());
			throw wrapper_exception();
		}

		pyobject_iterator_1 begin(prop.get_item, container);
		pyobject_iterator_1 end(prop.get_item, container, size);

		return np5::detail::approximate_l2(begin, end, degree);
	} else {
		PyErr_SetString(PyExc_ValueError,
			"Number of samples must be greater than 1");
			throw wrapper_exception();
	}
}


PyObject* build_result(mcore::linalg::vec&& v) noexcept {
	if (PyObject* result = PyTuple_New(v.dim())) {
		for (size_t i = 0; i < v.dim(); ++i) {
			if (PyObject* value = PyFloat_FromDouble(v(i))) {
				PyTuple_SetItem(result, i, value);
			} else {
				Py_DECREF(result);
				return nullptr;
			}
		}
		return result;
	} else
		return nullptr;
}

} // anonymous namespace


static PyObject*
approximate_l2_bind(PyObject* self, PyObject* args) {
	PyObject* arg1 = nullptr;
	PyObject* arg2 = nullptr;
	size_t degree = 0;

	try {
		mcore::linalg::vec V;
		if (PyArg_ParseTuple(args, "Ol", &arg1, &degree)) {
			if (PyTuple_CheckExact(arg1))
				V = std::move(build_l2_approximation(tuple_property, arg1, degree));
			else if (PyList_CheckExact(arg1))
				V = std::move(build_l2_approximation(list_property, arg1, degree));
			else {
				// Unknown type of arguments
				PyErr_SetString(PyExc_ValueError,
					"Unsupported type of arguments");
				return nullptr;
			}
		} else if (PyArg_ParseTuple(args, "OOl", &arg1, &arg2, &degree)) {
			assert(arg1 != nullptr);
			assert(arg2 != nullptr);

			auto par1 = parse_argument(arg1);
			auto par2 = parse_argument(arg2);

			if (std::get<0>(par1) == std::get<0>(par2)) {
				size_t const size = std::get<1>(par1);
				if (size) {
					if (size > 1) {
						if (size > degree) {
							// both sizes are equal, so we can continue
							pyobject_iterator_2 begin(
								arg1, std::get<0>(par1),
								arg2, std::get<0>(par2));
							pyobject_iterator_2 end(arg1, arg2, size);
							V = np5::detail::approximate_l2(begin, end, degree);
						} else {
							// Wrong combination of the degree and the data
							PyErr_SetString(PyExc_ValueError,
								"Wrong combination of the degree and the data");
							return nullptr;
						}
					} else {
						// Unknown type of arguments
						PyErr_SetString(PyExc_ValueError,
							"Number of samples in both sequences are equal to 1");
						return nullptr;
					}
				} else {
					// Unknown type of arguments
					PyErr_SetString(PyExc_ValueError,
						"Number of samples in both sequences are equal to zero");
					return nullptr;
				}
			}

		} else {
			// Unknown type of arguments
			PyErr_SetString(PyExc_ValueError,
				"Unsupported type of arguments");
			return nullptr;
		}
		return build_result(std::move(V));
	}

	catch (...) {
		return nullptr;
	}
}


static PyObject*
approximate_l1_bind(PyObject* self, PyObject* args) {
	PyObject* arg1 = nullptr;
	PyObject* arg2 = nullptr;
	size_t degree = 0;

	try {
		std::vector<point> pts;

		if (PyArg_ParseTuple(args, "Ol", &arg1, &degree))
			parse_data<point>(arg1).swap(pts);
		else if (PyArg_ParseTuple(args, "OOl", &arg1, &arg2, &degree))
			parse_data(arg1, arg2).swap(pts);
		else
			throw wrapper_exception(ERROR_UNKNOWN_ARGS);

		if (pts.size() < degree)
			throw wrapper_exception(ERROR_INSUFFICIENT_DATA);

		return build_result(np5::detail::approximate_l1(
			std::begin(pts), std::end(pts), degree));
	}

	catch(wrapper_exception const& e) {
		if (char const* m = e.message())
			PyErr_SetString(PyExc_ValueError, m);
		return nullptr;
	}

	catch (...) {
		return nullptr;
	}
}


static PyObject*
compute_nw_bandwidth(PyObject* self, PyObject* args) {
	PyObject* arg1 = nullptr;
	PyObject* arg2 = nullptr;
	std::vector<point> points;

	try {
		if (PyArg_ParseTuple(args, "O", &arg1))
			parse_data<point>(arg1).swap(points);
		else if (PyArg_ParseTuple(args, "OO", &arg1, &arg2))
			parse_data(arg1, arg2).swap(points);
		else
			throw wrapper_exception(ERROR_UNKNOWN_ARGS);

		double const bandwidth = np5::compute_band_nw(
			points.begin(), points.end());

		return PyFloat_FromDouble(bandwidth);
	}

	catch(wrapper_exception const& e) {
		if (char const* m = e.message())
			PyErr_SetString(PyExc_ValueError, m);
		return nullptr;
	}
}


static PyObject*
predict_nw_bind(PyObject* self, PyObject* args) {
	PyObject* p_data1 = nullptr;
	PyObject* p_data2 = nullptr;
	PyObject* p_x = nullptr;

	double x = std::numeric_limits<double>::quiet_NaN();
	double bw = std::numeric_limits<double>::quiet_NaN();

	std::vector<point> points;
	std::vector<double> xs;

	try {
		size_t const nargs = PyTuple_Size(args);
		switch (nargs) {
		case 3: {
			p_data1 = PyTuple_GetItem(args, 0);
			parse_data<point>(p_data1).swap(points);

			p_data2 = PyTuple_GetItem(args, 1);
			x = PyFloat_AsDouble(p_data2);
			if (PyErr_Occurred())
				parse_data<double_type>(p_data1).swap(xs);

			bw = PyFloat_AsDouble(PyTuple_GetItem(args, 2));
			if (PyErr_Occurred())
				throw wrapper_exception(ERROR_UNKNOWN_ARGS);
		}
		break;

		case 4: {
			p_data1 = PyTuple_GetItem(args, 0);
			p_data2 = PyTuple_GetItem(args, 1);
			parse_data(p_data1, p_data2).swap(points);

			PyObject* const a2 = PyTuple_GetItem(args, 2);
			x = PyFloat_AsDouble(a2);
			if (PyErr_Occurred()) {
				PyErr_Clear();
				parse_data<double_type>(a2).swap(xs);
			}

			bw = PyFloat_AsDouble(PyTuple_GetItem(args, 3));
			if (PyErr_Occurred()) {
				throw wrapper_exception(ERROR_UNKNOWN_ARGS);
			}
		}
		break;

		default:
			throw wrapper_exception(ERROR_UNKNOWN_ARGS);
		}

		if (xs.empty()) {
			double const value = np5::predict_nw(
				points.begin(), points.end(), x, bw);
			return PyFloat_FromDouble(value);
		} else {
			if (PyObject* result = PyTuple_New(xs.size())) {
				for (size_t i = 0; i < xs.size(); ++i) {
					double const value = np5::predict_nw(
					points.begin(), points.end(), xs[i], bw);
					if (PyObject* v = PyFloat_FromDouble(value)) {
						PyTuple_SetItem(result, i, v);
					} else {
						Py_DECREF(result);
						return nullptr;
					}
				}
				return result;
			} else
				return nullptr;
		}
	}

	catch (wrapper_exception const& e) {
		if (char const* m = e.message())
			PyErr_SetString(PyExc_ValueError, m);
		return nullptr;
	}

	catch(...) {
		return nullptr;
	}
}

char const compute_nw_bandwidth_descr[] =
"Compute optimal bandwidth for Nadaraya-Watson kernel regression";

static PyMethodDef SpamMethods[] = {
	{"approximate_l2", approximate_l2_bind, METH_VARARGS,
		"Creates L2 (LSQ) data apprixmation."},
	{"approximate_l1", approximate_l1_bind, METH_VARARGS,
		"Calculates L1 data approximation"},
	{"compute_bandwidth_NW", compute_nw_bandwidth, METH_VARARGS,
		compute_nw_bandwidth_descr},
	{"predict_NW", predict_nw_bind, METH_VARARGS, ""},
	{nullptr, nullptr, 0, nullptr}
};


PyMODINIT_FUNC
initregression(void) {
	(void) Py_InitModule("regression", SpamMethods);

	list_property.get_size = PyList_Size;
	list_property.get_item = PyList_GetItem;

	tuple_property.get_size = PyTuple_Size;
	tuple_property.get_item = PyTuple_GetItem;
}
