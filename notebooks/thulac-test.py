from numpy.f2py import use_rules
import thulac
from myhmm import make_cut



def _func_test():
    thu = thulac.thulac(seg_only=True)  #默认模式
    text = thu.cut("我爱北京天安门", text=True)  #进行一句话分词
    print(text)


thu = thulac.thulac(seg_only=True, user_dict='/home/luod/class/nlp/HanTokenization/datasets/training_vocab.txt')
def thulac_cut(sentence):
    text = thu.cut(sentence, text=True)
    return text.split()
    

def get_result():
    # make_cut(snow_cut, 'snow-train.txt')
    make_cut(thulac_cut, 'thulac-test.txt', train=False)


if __name__ == '__main__':
    get_result()