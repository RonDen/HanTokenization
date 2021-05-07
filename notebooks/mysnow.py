import sys
import gzip
import heapq
import marshal
import codecs
from math import log, exp
from functools import reduce


MIN_FLOAT = -3.14e100

def getz(r, nr):
    z = [2*nr[0]/r[1]]
    for i in range(len(nr)-2):
        z.append(2*nr[i+1]/(r[i+2]-r[i]))
    z.append(nr[-1]/(r[-1]-r[-2]))
    return z

def least_square(x, y): # y=a+bx
    meanx = sum(x)/len(x)
    meany = sum(y)/len(y)
    xy = sum((x[i]-meanx)*(y[i]-meany) for i in range(len(x)))
    square = sum((x[i]-meanx)**2 for i in range(len(x)))
    b = xy/square
    return (meany-b*meanx, b)


class BaseProb(object):

    def __init__(self):
        self.d = {}
        self.total = 0.0
        self.none = 0

    def exists(self, key):
        return key in self.d

    def getsum(self):
        return self.total

    def get(self, key):
        if not self.exists(key):
            return False, self.none
        return True, self.d[key]

    def freq(self, key):
        return float(self.get(key)[1])/self.total

    def samples(self):
        return self.d.keys()


class NormalProb(BaseProb):

    def add(self, key, value):
        if not self.exists(key):
            self.d[key] = 0
        self.d[key] += value
        self.total += value


class AddOneProb(BaseProb):

    def __init__(self):
        self.d = {}
        self.total = 0.0
        self.none = 1

    def add(self, key, value):
        self.total += value
        if not self.exists(key):
            self.d[key] = 1
            self.total += 1
        self.d[key] += value


def good_turing(dic):
    values = sorted(dic.values())
    r, nr, prob = [], [], []
    for v in values:
        if not r or r[-1] != v:
            r.append(v)
            nr.append(1)
        else:
            nr[-1] += 1
    rr = dict(map(lambda x:list(reversed(x)), enumerate(r)))
    total = reduce(lambda x, y:(x[0]*x[1]+y[0]*y[1], 1), zip(nr, r))[0]
    z = getz(r, nr)
    a, b = least_square(map(lambda x:log(x), r), map(lambda x:log(x), z))
    use_good_turing = False
    nr.append(exp(a+b*log(r[-1]+1)))
    for i in range(len(r)):
        good_turing = (r[i]+1)*(exp(b*(log(r[i]+1)-log(r[i]))))
        turing = (r[i]+1)*nr[i+1]/nr[i] if i+1<len(r) else good_turing
        diff = ((((r[i]+1)**2)/nr[i]*nr[i+1]/nr[i]*(1+nr[i+1]/nr[i]))**0.5)*1.65
        if not use_good_turing and abs(good_turing-turing)>diff:
            prob.append(turing)
        else:
            use_good_turing = True
            prob.append(good_turing)
    sump = reduce(lambda x, y:(x[0]*x[1]+y[0]*y[1], 1), zip(nr, prob))[0]
    for cnt, i in enumerate(prob):
        prob[cnt] = (1-nr[0]/total)*i/sump
    return nr[0]/total/total, dict(zip(dic.keys(), map(lambda x:prob[rr[x]], dic.values())))


class GoodTuringProb(BaseProb):
    def __init__(self):
        self.d = {}
        self.total = 0.0
        self.handled = False

    def add(self, key, value):
        if not self.exists(key):
            self.d[key] = 0
        self.d[key] += value

    def get(self, key):
        if not self.handled:
            self.handled = True
            tmp, self.d = good_turing(self.d)
            self.none = tmp
            self.total = sum(self.d.values())+0.0
        if not self.exists(key):
            return False, self.none
        return True, self.d[key]


class TnT(object):
    def __init__(self, N=1000):
        self.N = N
        self.l1 = 0.0
        self.l2 = 0.0
        self.l3 = 0.0
        self.status = set()
        self.wd = AddOneProb()
        self.eos = AddOneProb()
        self.eosd = AddOneProb()
        self.uni = NormalProb()
        self.bi = NormalProb()
        self.tri = NormalProb()
        self.word = {}
        self.trans = {}

    def save(self, fname, iszip=True):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, set):
                d[k] = list(v)
            elif hasattr(v, '__dict__'):
                d[k] = v.__dict__
            else:
                d[k] = v
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            marshal.dump(d, open(fname, 'wb'))
        else:
            f = gzip.open(fname, 'wb')
            f.write(marshal.dumps(d))
            f.close()

    def load(self, fname, iszip=True):
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            d = marshal.load(open(fname, 'rb'))
        else:
            try:
                f = gzip.open(fname, 'rb')
                d = marshal.loads(f.read())
            except IOError:
                f = open(fname, 'rb')
                d = marshal.loads(f.read())
            f.close()
        for k, v in d.items():
            if isinstance(self.__dict__[k], set):
                self.__dict__[k] = set(v)
            elif hasattr(self.__dict__[k], '__dict__'):
                self.__dict__[k].__dict__ = v
            else:
                self.__dict__[k] = v

    def tnt_div(self, v1, v2):
        if v2 == 0:
            return 0
        return float(v1)/v2

    def geteos(self, tag):
        tmp = self.eosd.get(tag)
        if not tmp[0]:
            return log(1.0/len(self.status))
        return log(self.eos.get((tag, 'EOS'))[1])-log(self.eosd.get(tag)[1])

    def train(self, data):
        for sentence in data:
            now = ['BOS', 'BOS']
            self.bi.add(('BOS', 'BOS'), 1)
            self.uni.add('BOS', 2)
            for word, tag in sentence:
                now.append(tag)
                self.status.add(tag)
                self.wd.add((tag, word), 1)
                self.eos.add(tuple(now[1:]), 1)
                self.eosd.add(tag, 1)
                self.uni.add(tag, 1)
                self.bi.add(tuple(now[1:]), 1)
                self.tri.add(tuple(now), 1)
                if word not in self.word:
                    self.word[word] = set()
                self.word[word].add(tag)
                now.pop(0)
            self.eos.add((now[-1], 'EOS'), 1)
        tl1 = 0.0
        tl2 = 0.0
        tl3 = 0.0
        for now in self.tri.samples():
            c3 = self.tnt_div(self.tri.get(now)[1]-1,
                              self.bi.get(now[:2])[1]-1)
            c2 = self.tnt_div(self.bi.get(now[1:])[1]-1,
                              self.uni.get(now[1])[1]-1)
            c1 = self.tnt_div(self.uni.get(now[2])[1]-1, self.uni.getsum()-1)
            if c3 >= c1 and c3 >= c2:
                tl3 += self.tri.get(now)[1]
            elif c2 >= c1 and c2 >= c3:
                tl2 += self.tri.get(now)[1]
            elif c1 >= c2 and c1 >= c3:
                tl1 += self.tri.get(now)[1]
        self.l1 = float(tl1)/(tl1+tl2+tl3)
        self.l2 = float(tl2)/(tl1+tl2+tl3)
        self.l3 = float(tl3)/(tl1+tl2+tl3)
        for s1 in self.status | set(('BOS',)):
            for s2 in self.status | set(('BOS',)):
                for s3 in self.status:
                    uni = self.l1*self.uni.freq(s3)
                    bi = self.tnt_div(self.l2*self.bi.get((s2, s3))[1],
                                      self.uni.get(s2)[1])
                    tri = self.tnt_div(self.l3*self.tri.get((s1, s2, s3))[1],
                                       self.bi.get((s1, s2))[1])
                    if uni+bi+tri == 0:
                        self.trans[(s1, s2, s3)] = MIN_FLOAT
                    else:
                        self.trans[(s1, s2, s3)] = log(uni + bi + tri)

    def tag(self, data):
        now = [(('BOS', 'BOS'), 0.0, [])]
        for w in data:
            stage = {}
            samples = self.status
            if w in self.word:
                samples = self.word[w]
            for s in samples:
                wd = log(self.wd.get((s, w))[1])-log(self.uni.get(s)[1])
                for pre in now:
                    p = pre[1]+wd+self.trans[(pre[0][0], pre[0][1], s)]
                    if (pre[0][1], s) not in stage or p > stage[(pre[0][1],
                                                                 s)][0]:
                        stage[(pre[0][1], s)] = (p, pre[2]+[s])
            stage = list(map(lambda x: (x[0], x[1][0], x[1][1]), stage.items()))
            now = heapq.nlargest(self.N, stage, key=lambda x: x[1])
        now = heapq.nlargest(1, stage, key=lambda x: x[1]+self.geteos(x[0][1]))
        return zip(data, now[0][2])


class Seg(object):

    def __init__(self, name='tnt'):
        if name == 'tnt':
            self.segger = TnT()
        # else:
        #     self.segger = CharacterBasedGenerativeModel()

    def save(self, fname, iszip=True):
        self.segger.save(fname, iszip)

    def load(self, fname, iszip=True):
        self.segger.load(fname, iszip)

    def train(self, fname):
        fr = codecs.open(fname, 'r', 'utf-8')
        data = []
        for i in fr:
            line = i.strip()
            if not line:
                continue
            tmp = map(lambda x: x.split('/'), line.split())
            data.append(tmp)
        fr.close()
        self.segger.train(data)

    def seg(self, sentence):
        ret = self.segger.tag(sentence)
        tmp = ''
        for i in ret:
            if i[1] == 'E':
                yield tmp+i[0]
                tmp = ''
            elif i[1] == 'B' or i[1] == 'S':
                if tmp:
                    yield tmp
                tmp = i[0]
            else:
                tmp += i[0]
        if tmp:
            yield tmp

from myhmm import make_cut
import re
re_zh = re.compile('([\u4E00-\u9FA5]+)')

tntseger = Seg()
tntseger.train('pku_training.tagging.utf8')

def seg(sent):
    words = []
    for s in re_zh.split(sent):
        s = s.strip()
        if not s:
            continue
        if re_zh.match(s):
            words += tntseger.seg(s)
        else:
            for word in s.split():
                word = word.strip()
                if word:
                    words.append(word)
    return words

def get_result():
    make_cut(seg, 'snow-trained.txt')
    

  
get_result()
