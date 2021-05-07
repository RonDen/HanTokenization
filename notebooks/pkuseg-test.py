import pkuseg
from myhmm import make_cut


def _func_test():
    seg = pkuseg.pkuseg()           # 以默认配置加载模型
    text = seg.cut('我爱北京天安门')  # 进行分词
    print(text)


# _func_test()

# seg = pkuseg.pkuseg(user_dict='/home/luod/class/nlp/HanTokenization/datasets/training_vocab.txt', model_name='models/pku')
# def pkuseg_cut(sentence):
#     return seg.cut(sentence)


# def get_result():
#     make_cut(pkuseg_cut, 'pkuseg-test.txt', train=False)


def train():
    pkuseg.train(
        trainFile='/home/luod/class/nlp/HanTokenization/datasets/training.txt',
        testFile='/home/luod/class/nlp/HanTokenization/datasets/test.txt',
        savedir='./models/pku')

def test():
    pkuseg.test()
    
    pass

if __name__ == '__main__':
    # get_result()
    train()

    # test()

