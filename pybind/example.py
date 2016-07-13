#!/bin/python

try:
    import regression as r
except:
    from sys import exit
    print "Please compile the library"
    exit(0)

import numpy as np


def print_to_gnuplot(ts, evs, avs):
    data = sorted((x, y, z) for x, y, z in zip(ts, evs, avs))
    s = ["plot '-' wit li, '-' wi li"]
    for t, x, _ in data:
        s.append('%s %s' % (t, x))
    s.append('e')
    for t, _, x in data:
        s.append('%s %s' % (t, x))
    s.append('e')
    s.append('pause(-1)')
    return '\n'.join(s)


def run_kernel():
    u0, u1, du = 0, 8, 0.05
    xs = np.arange(u0, u1, du)
    ys = np.sin(xs)

    print 'Using Nadaraya-Watson regression for clean data'
    print 'Computed bandwidth = ',
    xs, ys = tuple(xs), tuple(ys)
    bw = r.compute_bandwidth_NW(xs, ys)
    print bw

    print 'Prediction, max deviation = ',
    ts = tuple(np.random.uniform(u0, u1, 2000))
    ws = r.predict_NW(xs, ys, ts, bw)
    print max(np.abs(np.sin(ts) - np.array(ws)))

    with open('graph.gpl', 'w') as fd:
        fd.write(print_to_gnuplot(ts, np.sin(ts), ws))




if __name__ == '__main__':
    run_kernel()
