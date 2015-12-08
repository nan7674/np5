import regression

def build_data():
	F = lambda x: 1 + 0.5 * x
	x, ds = 0, []
	while x < 1:
		ds.append((x, F(x)))
		x += 0.02
	return ds

def f1():
	print regression.approximate_l2(tuple(build_data()), 1)
	
def f2():
	print regression.approximate_l2(build_data(), 1)

if __name__ == '__main__':
	f1()
	f2()
