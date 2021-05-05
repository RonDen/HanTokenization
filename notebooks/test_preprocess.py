from preprocess import preprocess, han_com, UNK
from myhmm import hmm_cut


def _test_preprocess():
    sentence = '迈向充满希望的新世纪——一九九八年新年讲话（附图片1张）２００％２０００．０００％北京十二月三十一日电（’９８'
    sentence = '同胞们、朋友们、女士们、先生们：在１９９８年来临之际，我十分高兴地通过中央人民广播电台、中国国际广播电台和中央电视台，向全国各族人民，向香港特别行政区同胞、澳门和台湾同胞、海外侨胞，向世界各国的朋友们，致以诚挚的问候和良好的祝愿！'
    sens, rec = preprocess(sentence=sentence)
    print(sentence)
    print(sens)
    print(rec)
    res, idx = [], 0
    le, ri = 0, 0
    while ri < len(sens):
        if sens[ri] == UNK:
            if le < ri:
                res += hmm_cut(sens[le: ri])
            le = ri + 1
            res += [rec[idx]]
            idx += 1
        ri += 1
    if ri == len(sens) and sens[-1] != UNK:
        res += hmm_cut(sens[le:])
    print(res)
    print("".join(res))
    assert "".join(res) == sentence



if __name__ == '__main__':
    # print(list(hmm_cut('我十分高兴地通过中央人民广播电台、中国国际广播电台和中央电视台，向全国各族人民，向香港特别行政区同胞')))
    # print(list(hmm_cut('在@来临之际，我十分高兴地通过中央人民广播电台、中国国际广播电台和中央电视台，向全国各族人民，向香港特别行政区同胞、澳门和台湾同胞、海外侨胞，向世界各国的朋友们，致以诚挚的问候和良好的祝愿！')))
    _test_preprocess()