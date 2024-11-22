from typing import Optional
from typing_extensions import TypedDict

import dill
import numpy as np

from medcat2.storage.serialisables import AbstractSerialisable


WordDescriptor = TypedDict('WordDescriptor',
                           {'vector': Optional[np.ndarray],
                            'count': int, 'index': int})


class Vocab(AbstractSerialisable):
    """Vocabulary used to store word embeddings for context similarity
    calculation. Also used by the spell checker - but not for fixing the
    spelling only for checking is something correct.

    Properties:
        vocab (dict[str, WordDescriptor]):
            Map from word to attributes, e.g. {'house':
                {'vector': <np.array>, 'count': <int>, ...}, ...}
        index2word (dict[int, str]):
            From word to an index - used for negative sampling
        vec_index2word (dict):
            Same as index2word but only words that have vectors
    """
    def __init__(self) -> None:
        super().__init__()
        self.vocab: dict[str, WordDescriptor] = {}
        self.index2word: dict[int, str] = {}
        self.vec_index2word: dict[int, str] = {}
        self.cum_probs: np.ndarray = np.array([])

    def inc_or_add(self, word: str, cnt: int = 1,
                   vec: Optional[np.ndarray] = None) -> None:
        """Add a word or incrase its count.

        Args:
            word(str):
                Word to be added
            cnt(int):
                By how much should the count be increased, or to what
                should it be set if a new word. (Default value = 1)
            vec(Optional[np.ndarray]):
                Word vector (Default value = None)
        """
        if word not in self.vocab:
            self.add_word(word, cnt, vec)
        else:
            self.inc_wc(word, cnt)

    def remove_all_vectors(self) -> None:
        """Remove all stored vector representations."""
        self.vec_index2word = {}

        for word in self.vocab:
            self.vocab[word]['vector'] = None

    def remove_words_below_cnt(self, cnt: int) -> None:
        """Remove all words with frequency below cnt.

        Args:
            cnt(int):
                Word count limit.
        """
        for word in list(self.vocab.keys()):
            if self.vocab[word]['count'] < cnt:
                del self.vocab[word]

        # Rebuild index2word and vec_index2word
        self._rebuild_index()

    def _rebuild_index(self):
        self.index2word = {}
        self.vec_index2word = {}
        for word, word_info in self.vocab.items():
            ind = len(self.index2word)
            self.index2word[ind] = word
            word_info['index'] = ind

            if word_info['vector'] is not None:
                self.vec_index2word[ind] = word

    def inc_wc(self, word: str, cnt: int = 1) -> None:
        """Incraese word count by cnt.

        Args:
            word(str):
                For which word to increase the count
            cnt(int):
                By how muhc to incrase the count (Default value = 1)
        """
        self.item(word)['count'] += cnt

    def add_vec(self, word: str, vec: np.ndarray) -> None:
        """Add vector to a word.

        Args:
            word(str):
                To which word to add the vector.
            vec(np.ndarray):
                The vector to add.
        """
        self.vocab[word]['vector'] = vec

        ind = self.vocab[word]['index']
        if ind not in self.vec_index2word:
            self.vec_index2word[ind] = word

    def reset_counts(self, cnt: int = 1) -> None:
        """Reset the count for all word to cnt.

        Args:
            cnt(int):
                New count for all words in the vocab. (Default value = 1)
        """
        for word in self.vocab.keys():
            self.vocab[word]['count'] = cnt

    def update_counts(self, tokens: list[str]) -> None:
        """Given a list of tokens update counts for words in the vocab.

        Args:
            tokens(list[str]):
                Usually a large block of text split into tokens/words.
        """
        for token in tokens:
            if token in self:
                self.inc_wc(token, 1)

    def add_word(self, word: str, cnt: int = 1,
                 vec: Optional[np.ndarray] = None,
                 replace: bool = True) -> None:
        """Add a word to the vocabulary

        Args:
            word (str):
                The word to be added, it should be lemmatized and lowercased
            cnt (int):
                Count of this word in your dataset (Default value = 1)
            vec (Optional[np.ndarray]):
                The vector representation of the word (Default value = None)
            replace (bool):
                Will replace old vector representation (Default value = True)
        """
        if word not in self.vocab:
            # NOTE: If one were to manually remove a word, this could have
            #       issues, but the Vocab should - in general - be pretty
            #       stable, so shouldn't be an issue
            ind = len(self.index2word)
            self.index2word[ind] = word
            item: WordDescriptor = {'vector': vec, 'count': cnt, 'index': ind}
            self.vocab[word] = item

            if vec is not None:
                self.vec_index2word[ind] = word
        elif replace and vec is not None:
            word_info = self.vocab[word]
            word_info['vector'] = vec
            word_info['count'] = cnt

            # If this word didn't have a vector before
            ind = word_info['index']
            if ind not in self.vec_index2word:
                self.vec_index2word[ind] = word

    def add_words(self, path: str, replace: bool = True) -> None:
        """Adds words to the vocab from a file, the file
        is required to have the following format (vec being optional):
            <word>\t<cnt>[\t<vec_space_separated>]

        e.g. one line: the word house with 3 dimensional vectors
            house   34444   0.3232 0.123213 1.231231

        Args:
            path(str):
                path to the file with words and vectors
            replace(bool):
                existing words in the vocabulary will be replaced.
                Defaults to True.
        """
        with open(path) as f:
            for line in f:
                parts = line.split("\t")
                word = parts[0]
                cnt = int(parts[1].strip())
                vec = None
                if len(parts) == 3:
                    floats = [float(x) for x in parts[2].strip().split(" ")]
                    vec = np.array(floats)

                self.add_word(word, cnt, vec, replace)

    def init_cumsums(self) -> None:
        """Initialise cumulative sums.

        This is in place of the unigram table. But similarly to it, this
        approach allows generating a list of indices that match the
        probabilistic distribution expected as per the word counts of each
        word.
        """
        raw_freqs = []

        words = list(self.vec_index2word.values())
        for word in words:
            raw_freqs.append(self[word])

        freqs = np.array(raw_freqs) ** (3/4)
        freqs /= freqs.sum()

        self.cum_probs = np.cumsum(freqs)

    def get_negative_samples(self, n: int = 6,
                             ignore_punct_and_num: bool = False) -> list[int]:
        """Get N negative samples.

        Args:
            n (int):
                How many words to return (Default value = 6)
            ignore_punct_and_num (bool):
                Whether to ignore punctuation and numbers. Defaults to False.

        Raises:
            Exception: If no unigram table is present.

        Returns:
            list[int]:
                Indices for words in this vocabulary.
        """
        if len(self.cum_probs) == 0:
            self.init_cumsums()
        random_vals = np.random.rand(n)
        inds: list[int] = np.searchsorted(self.cum_probs,
                                          random_vals).tolist()

        if ignore_punct_and_num:
            # Do not return anything that does not have letters in it
            return [ind for ind in inds
                    if self.index2word[ind].upper().isupper()]
        return inds

    def get_vectors(self, indices: list[int]) -> list[np.ndarray]:
        return [self.vec(self.index2word[ind])  # type: ignore
                for ind in indices if ind in self.vec_index2word]

    def __getitem__(self, word: str) -> int:
        return self.count(word)

    def vec(self, word: str) -> Optional[np.ndarray]:
        return self.vocab[word]['vector']

    def count(self, word: str) -> int:
        return self.vocab[word]['count']

    def item(self, word: str) -> WordDescriptor:
        return self.vocab[word]

    def __contains__(self, word: str) -> bool:
        if word in self.vocab:
            return True

        return False

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, 'rb') as f:
            vocab: Vocab = dill.load(f)
        if not hasattr(vocab, 'cum_probs'):
            # NOTE: this is not too expensive, only around 0.05s
            vocab.init_cumsums()
        if hasattr(vocab, 'unigram_table'):
            del vocab.unigram_table
        return vocab
