import dill

from medcat2.vocab import Vocab


def get_vocab_from_old(old_path: str) -> Vocab:
    with open(old_path, 'rb') as f:
        data = dill.load(f)
    v = Vocab()
    v.vocab = {
        k: {'vector': v['vec'], 'count': v['cnt'], 'index': v['ind']}
        for k, v in data['vocab'].items()}
    v.index2word = data['index2word']
    v.make_unigram_table()
    return v
