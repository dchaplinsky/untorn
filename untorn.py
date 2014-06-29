import sys
import cv2
import math
import pylab
import numpy as np
from random import randint

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
TABULATION_STEPS = 200


def angle(a, b, c):
    ba = (b - a)[0]
    cb = (c - b)[0]

    ang1 = math.atan2(ba[0], ba[1])
    ang2 = math.atan2(cb[0], cb[1])

    return ang2 - ang1


def cv2color2rgb(cv2c):
    return map(lambda x: x / 255., cv2c[::-1])


def smooth(seq, window=5):
    assert(window % 2 == 1)
    half_win = int(window / 2)

    def get(idx):
        if idx < 0:
            return seq[0]
        if idx > len(seq) - 1:
            return seq[-1]

        return seq[idx]

    def apply_window(idx):
        return sum(get(x)[1] for x in xrange(idx - half_win,
                                             idx + half_win + 1)) / window

    return [[seq[x][0], apply_window(x)] for x in xrange(len(seq))]


class Edge(object):
    def __init__(self, fragment_no, edge_no, path):
        self.fragment_no = fragment_no
        self.edge_no = edge_no
        self.path = path
        self.used = False

        self.edge_len = cv2.arcLength(np.array([self.path[0], self.path[-1]]),
                                      False)
        self.path_len = cv2.arcLength(self.path, False)

        cp1 = self.path[0]
        cp2 = self.path[-1]
        self.env = [[1., 0.]]

        for e in self.path[1:-1]:
            alpha = angle(cp1, cp2, e)
            l = (cp2 - e)[0]
            l = math.sqrt(l[0] ** 2 + l[1] ** 2)
            self.env.append([-l * math.cos(alpha) / self.edge_len,
                             l * math.sin(alpha)])

        self.env.append([0., 0.])
        self.env.reverse()

        self.env = smooth(self.env, 5)
        self.is_paper_edge = abs(self.edge_len - self.path_len) < 1

    def interpolate_env(self, x):
        assert x >= 0 and x <= 1

        x0 = 0
        x1 = 0
        y0 = 0
        y1 = 0
        for i, (xi, yi) in enumerate(self.env):
            if xi >= x:
                x0, y0 = self.env[i - 1]
                x1, y1 = self.env[i]
                break

        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def tabulate_env(self, steps=100):
        return np.array(map(self.interpolate_env,
                        np.array(range(steps + 1)) / float(steps)))

    def __repr__(self):
        return "%s:%s" % (self.fragment_no, self.edge_no)


class Fragment(object):
    def __init__(self, fragment, fragment_no, edges):
        self.fragment = fragment
        self.fragment_no = fragment_no
        self.edges = edges


def find_match(edge, edges, top_n=3):
    def is_possible_match(x):
        ratio = x.edge_len / edge.edge_len
        if ratio < 1:
            ratio = 1 / ratio

        return (not x.used and x.fragment_no != edge.fragment_no and
                ratio < 1.1)

    candidates = filter(is_possible_match, edges)
    if not candidates:
        return []

    env = edge.tabulate_env(TABULATION_STEPS)

    candidates_with_metric = []
    for c in candidates:
        c_env = c.tabulate_env(TABULATION_STEPS)

        NEED = sum(abs(env + c_env[::-1])) / max(c.path_len, edge.path_len)
        candidates_with_metric.append([c, NEED])

    return sorted(candidates_with_metric, key=lambda x: x[1])[:top_n]


if __name__ == '__main__':
    fname = sys.argv[1] if len(sys.argv) > 1 else "src/simple.png"

    src = cv2.imread(fname)

    src_binary = cv2.bitwise_not(cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY))
    contours, _ = cv2.findContours(src_binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(src, contours, -1, (255, 0, 0), 2)
    fragments = []
    all_edges = []

    for k, cnt in enumerate(contours):
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        sides = []

        for i in xrange(len(approx)):
            ang = (angle(approx[i - 2], approx[i - 1],
                         approx[i])) * 180 / math.pi

            ang = (ang + 360) % 360
            if ang > 180:
                ang = 360 - ang

            if abs(ang - 90) < 40:
                cv2.circle(src, tuple(approx[i - 1][0]), 3, (0, 0, 255), -1)
                sides.append(approx[i - 1])

        cnt = cnt.tolist()
        indx = []
        for s in sides:
            indx.append(cnt.index(s.tolist()))

        edges = []
        for i in xrange(len(indx)):
            if indx[i - 1] < indx[i]:
                arr = np.array(cnt[indx[i - 1]:indx[i] + 1])
            else:
                arr = np.array(cnt[indx[i - 1]:len(cnt)] + cnt[:indx[i] + 1])

            edges.append(Edge(k, i, arr))

            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            cv2.polylines(src, [arr], False, color, 2)

            # pylab.plot(edges[-1].env,
            #            color=cv2color2rgb(color), label="%s:%s" % (k, i))

            pylab.plot(edges[-1].tabulate_env(TABULATION_STEPS),
                       color=cv2color2rgb(color), label="%s:%s" % (k, i))

        fragments.append(Fragment(cnt, k, edges))
        all_edges += edges

    for edge in all_edges:
        arr = edge.path
        coords = arr[len(arr) / 2][0]
        text_size, _ = cv2.getTextSize(str(edge), FONT_FACE, FONT_SCALE, 1)

        # coords -= np.array(text_size) / 2
        cv2.putText(src, str(edge), tuple(coords + [1, 0]), FONT_FACE,
                    FONT_SCALE, (0, 0, 0))

        cv2.putText(src, str(edge), tuple(coords), FONT_FACE, FONT_SCALE,
                    (0, 0, 255))

    print("Matches for %s:" % all_edges[2])
    print(find_match(all_edges[2], all_edges))
    # pylab.legend(loc='best', shadow=True)
    cv2.imshow("src", src)
    pylab.show()
    cv2.waitKey()
