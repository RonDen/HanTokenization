from snownlp import SnowNLP
from myhmm import make_cut


def _func_test():
    s = SnowNLP(u'这个东西真心很赞')

    print(s.words)         # [u'这个', u'东西', u'真心',
                    #  u'很', u'赞']

    print(s.tags)          # [(u'这个', u'r'), (u'东西', u'n'),
                    #  (u'真心', u'd'), (u'很', u'd'),
                    #  (u'赞', u'Vg')]

    print(s.sentiments)    # 0.9769663402895832 positive的概率

    print(s.pinyin)        # [u'zhe', u'ge', u'dong', u'xi',
                    #  u'zhen', u'xin', u'hen', u'zan']

    s = SnowNLP(u'「繁體字」「繁體中文」的叫法在臺灣亦很常見。')

    print(s.han)           # u'「繁体字」「繁体中文」的叫法
                    # 在台湾亦很常见。'

    text = u'''
    自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
    它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
    自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
    因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，
    所以它与语言学的研究有着密切的联系，但又有重要的区别。
    自然语言处理并不是一般地研究自然语言，
    而在于研制能有效地实现自然语言通信的计算机系统，
    特别是其中的软件系统。因而它是计算机科学的一部分。
    '''

    s = SnowNLP(text)

    print(s.keywords(3))	# [u'语言', u'自然', u'计算机']

    print(s.summary(3))	# [u'因而它是计算机科学的一部分',
                    #  u'自然语言处理是一门融语言学、计算机科学、
                    #	 数学于一体的科学',
                    #  u'自然语言处理是计算机科学领域与人工智能
                    #	 领域中的一个重要方向']
    print(s.sentences)

    s = SnowNLP([[u'这篇', u'文章'],
                [u'那篇', u'论文'],
                [u'这个']])
    print(s.tf)
    print(s.idf)
    print(s.sim([u'文章']))
    # [0.3756070762985226, 0, 0]

# _func_test()

def snow_cut(sentence):
    return SnowNLP(sentence).words


def get_result():
    # make_cut(snow_cut, 'snow-train.txt')
    make_cut(snow_cut, 'snow-test.txt')



if __name__ == '__main__':
    get_result()