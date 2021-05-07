import thulac
from myhmm import make_cut, train_set_split



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


def make_train_set():
    with open("pku-thulac-train.txt", 'w', encoding='utf8') as f:
        for word_list in train_set_split:
            line = "/ ".join(word_list)
            f.write(line + '/\n')

    pass

if __name__ == '__main__':
    # make_train_set()
    get_result()
