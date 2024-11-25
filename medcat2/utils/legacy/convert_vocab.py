import dill

from medcat2.vocab import Vocab


def get_vocab_from_old(old_path: str) -> Vocab:
    with open(old_path, 'rb') as f:
        data = dill.load(f)
    v = Vocab()
    for word, word_data in data['vocab'].items():
        vec = word_data['vec']
        if len(vec) != 7:
            print("WORD HAS WRONG VEC:", word, "len of vec", len(vec))
            print("Fixing just now!")
            vec = vec[:7]
        v.add_word(word, cnt=word_data['cnt'], vec=vec,
                   replace=True)
    v.init_cumsums()
    return v
