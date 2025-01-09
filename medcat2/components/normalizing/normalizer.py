from typing import Optional, Iterable, Iterator
import re

from medcat2.tokenizing.tokens import MutableDocument
from medcat2.tokenizing.tokenizers import BaseTokenizer
from medcat2.config.config import Config
from medcat2.vocab import Vocab
from medcat2.components.types import CoreComponentType, AbstractCoreComponent


CONTAINS_NUMBER = re.compile('[0-9]+')


class BasicSpellChecker:

    def __init__(self, cdb_vocab: dict[str, int], config: Config,
                 data_vocab: Optional[Vocab] = None):
        self.vocab = cdb_vocab
        self.config = config
        self.data_vocab = data_vocab

    def P(self, word: str) -> float:
        """Probability of `word`.

        Args:
            word (str): The word in question.

        Returns:
            float: The probability.
        """
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
        cnt = self.vocab.get(word, 0)
        if cnt != 0:
            return -1 / cnt
        else:
            return 0

    def __contains__(self, word):
        if word in self.vocab:
            return True
        elif self.data_vocab is not None and word in self.data_vocab:
            return False
        else:
            return False

    def fix(self, word: str) -> Optional[str]:
        """Most probable spelling correction for word.

        Args:
            word (str): The word.

        Returns:
            Optional[str]: Fixed word, or None if no fixes were applied.
        """
        fix = max(self.candidates(word), key=self.P)
        if fix != word:
            return fix
        else:
            return None

    def candidates(self, word: str) -> Iterable[str]:
        """Generate possible spelling corrections for word.

        Args:
            word (str): The word.

        Returns:
            Iterable[str]: The list of candidate words.
        """
        if self.config.general.spell_check_deep:
            # This will check a two letter edit distance
            return (self.known([word]) or
                    self.known(self.edits1(word)) or
                    self.known(self.edits2(word)) or
                    [word])
        else:
            # Will check only one letter edit distance
            return (self.known([word]) or
                    self.known(self.edits1(word)) or
                    [word])

    def known(self, words: Iterable[str]) -> set[str]:
        """The subset of `words` that appear in the dictionary of WORDS.

        Args:
            words (Iterable[str]): The words.

        Returns:
            Set[str]: The set of candidates.
        """
        return set(w for w in words if w in self.vocab)

    def edits1(self, word: str) -> set[str]:
        """All edits that are one edit away from `word`.

        Args:
            word (str): The word.

        Returns:
            Set[str]: The set of all edits
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'

        if self.config.general.diacritics:
            letters += 'àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'

        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes: list[str] = []
        transposes: list[str] = []
        replaces: list[str] = []
        inserts: list[str] = []
        for L, R in splits:
            if R:
                deletes.append(L + R[1:])
            if len(R) > 1:
                transposes.append(L + R[1] + R[0] + R[2:])
            if R:
                replaces.extend(L + c + R[1:] for c in letters)
            inserts.extend([L + c + R for c in letters])
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str) -> Iterator[str]:
        """All edits that are two edits away from `word`.

        Args:
            word (str): The word to start from.

        Returns:
            Iterator[str]: All 2-away edits.
        """
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def edits3(self, word):
        """All edits that are two edits away from `word`."""  # noqa
        # Do d3 edits
        raise ValueError("No implementation")


class TokenNormalizer(AbstractCoreComponent):
    """Will normalize all tokens in a spacy document.    """
    name = 'token_normalizer'

    # Override
    def __init__(self, nlp: BaseTokenizer, config: Config,
                 cdb_vocab: dict[str, int],
                 data_vocab: Optional[Vocab] = None):
        self.config = config
        self.spell_checker = BasicSpellChecker(cdb_vocab, config, data_vocab)
        self.nlp = nlp

    def get_type(self) -> CoreComponentType:
        return CoreComponentType.token_normalizing

    # Override
    def __call__(self, doc: MutableDocument):
        # avoid accessing all these in loop
        spell_check_limit = self.config.general.spell_check_len_limit
        min_len_normalizer = self.config.preprocessing.min_len_normalize
        do_not_normalize = self.config.preprocessing.do_not_normalize
        perform_spell_check = self.config.general.spell_check
        for token in doc:
            if len(token.base.lower) < min_len_normalizer:
                token.norm = token.base.lower
            elif (do_not_normalize and
                    token.tag is not None and
                    token.tag in do_not_normalize):
                token.norm = token.base.lower
            elif token.lemma == '-PRON-':
                token.norm = token.lemma
                token.to_skip = True
            else:
                token.norm = token.lemma.lower()

            if perform_spell_check:
                # Fix the token if necessary
                if (len(token.base.text) >= spell_check_limit and
                        not token.is_punctuation and self.spell_checker and
                        token.base.lower not in self.spell_checker and
                        not CONTAINS_NUMBER.search(token.base.lower)):
                    fix = self.spell_checker.fix(token.base.lower)
                    if fix is not None:
                        tmp = self.nlp(fix)[0]
                        if len(token.base.lower) < min_len_normalizer:
                            token.norm = tmp.base.lower
                        else:
                            token.norm = tmp.lemma.lower()
        return doc


def set_default_args(config: Config, tokenizer: BaseTokenizer,
                     cdb_vocab: dict[str, int], vocab: Optional[Vocab]
                     ) -> None:
    config.components.token_normalizing.init_args = [
        tokenizer, config, cdb_vocab, vocab
    ]
