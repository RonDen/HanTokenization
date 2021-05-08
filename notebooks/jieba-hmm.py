from myhmm import cut
from jieba.analyse.analyzer



def __cut_DAG(self, sentence):
    DAG = self.get_DAG(sentence)
    route = {}
    self.calc(sentence, DAG, route)
    x = 0
    buf = ''
    N = len(sentence)
    while x < N:
        y = route[x][1] + 1
        l_word = sentence[x:y]
        if y - x == 1:
            buf += l_word
        else:
            if buf:
                if len(buf) == 1:
                    yield buf
                    buf = ''
                else:
                    if not self.FREQ.get(buf):
                        recognized = cut(buf)
                        for t in recognized:
                            yield t
                    else:
                        for elem in buf:
                            yield elem
                    buf = ''
            yield l_word
        x = y

    if buf:
        if len(buf) == 1:
            yield buf
        elif not self.FREQ.get(buf):
            recognized = cut(buf)
            for t in recognized:
                yield t
        else:
            for elem in buf:
                yield elem

