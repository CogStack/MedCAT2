from typing import Optional
import logging
from datetime import datetime

from pydantic import BaseModel, Field

from medcat2.utils.defaults import workers
from medcat2.utils.envsnapshot import Environment, get_environment_info


logger = logging.getLogger(__name__)


class CoreComponentConfig(BaseModel):
    comp_name: str = 'default'
    """The name of the component.

    If a custom implementation is required, it needs to be registered
    using `medcat2.components.types.register_core_component(
            <core component type>, <component name>, <implementing class>)
    By default, only the 'default' component is registered.
    """
    init_args: list = Field(default_factory=list, exclude=True)
    """These are the positional arguments required to construct the component.

    For default components, these will be automatically filled. However, if a
    custom component is used, these would need to be set manually.
    """
    init_kwargs: dict = Field(default_factory=dict, exclude=True)
    """These are the keyword arguments required to construct the component.

    For default components, these will be automatically filled. However, if a
    custom component is used, these would need to be set manually.
    """


class NLPConfig(BaseModel):
    disabled_components: list = ['ner', 'parser', 'vectors', 'textcat',
                                 'entity_linker', 'sentencizer',
                                 'entity_ruler', 'merge_noun_chunks',
                                 'merge_entities', 'merge_subtokens']
    """The list of components that will be disabled for the NLP.

    NB! For these changes to take effect, the pipe would need to be recreated.
    """
    provider: str = 'spacy'
    """The NLP provider.

    Currently only spacy is natively supported.

    NB! For these changes to take effect, the pipe would need to be recreated.
    """
    modelname: str = 'en_core_web_md'
    """What model will be used for tokenization.

    NB! For these changes to take effect, the pipe would need to be recreated.
    """
    init_args: list = Field(default_factory=list, exclude=True)
    """These are the positional arguments required to construct the component.

    For default components, these will be automatically filled. However, if a
    custom component is used, these would need to be set manually.
    """
    init_kwargs: dict = Field(default_factory=dict, exclude=True)
    """These are the keyword arguments required to construct the component.

    For default components, these will be automatically filled. However, if a
    custom component is used, these would need to be set manually.
    """

    # NOTE: this will allow for more config entries
    #       since we don't know what other implementations may require
    class Config:
        extra = 'allow'
        validate_assignment = True


class General(BaseModel):
    """The general part of the config"""
    nlp: NLPConfig = NLPConfig()
    # checkpoint: CheckPoint = CheckPoint()
    # usage_monitor = UsageMonitor()
    """Checkpointing config"""
    log_level: int = logging.INFO
    """Logging config for everything | 'tagger' can be disabled,
    but will cause a drop in performance"""
    log_format: str = '%(levelname)s:%(name)s: %(message)s'
    log_path: str = './medcat.log'
    separator: str = '~'
    """Separator that will be used to merge tokens of a name.
    Once a CDB is built this should always stay the same."""
    spell_check: bool = True
    """Should we check spelling - note that this makes things much slower,
    use only if necessary. The only thing necessary for the spell checker
    to work is vocab.dat and cdb.dat built with concepts in the respective
    language."""
    diacritics: bool = False
    """Should we process diacritics - for languages other than English,
    symbols such as 'é, ë, ö' can be relevant. Note that this makes
    spell_check slower."""
    spell_check_deep: bool = False
    """If True the spell checker will try harder to find mistakes,
    this can slow down things drastically."""
    spell_check_len_limit: int = 7
    """Spelling will not be checked for words with length less than this"""
    show_nested_entities: bool = False
    """If set to True functions like get_entities and get_json will return
    nested_entities and overlaps"""
    full_unlink: bool = False
    """When unlinking a name from a concept should we do full_unlink
    (means unlink a name from all concepts, not just the one in question)"""
    workers: int = workers()
    """Number of workers used by a parallelizable pipeline component"""
    make_pretty_labels: Optional[str] = None
    """Should the labels of entities (shown in displacy) be pretty
    or just 'concept'. Slows down the annotation pipeline
    should not be used when annotating millions of documents.
    If `None` it will be the string "concept", if `short` it will be CUI,
    if `long` it will be CUI | Name | Confidence"""
    map_cui_to_group: bool = False
    """If the cdb.addl_info['cui2group'] is provided and this option enabled,
    each CUI will be maped to the group"""
    simple_hash: bool = False
    """Whether to use a simple hash.

    NOTE: While using a simple hash is faster at save time, it is less
    reliable due to not taking into account all the details of the changes."""


class LinkingFilters(BaseModel):
    """These describe the linking filters used alongside the model.

    When no CUIs nor exlcuded CUIs are specified (the sets are empty),
    all CUIs are accepted.
    If there are CUIs specified then only those will be accepted.
    If there are excluded CUIs specified, they are excluded.

    In some cases, there are extra filters as well as MedCATtrainer (MCT)
    export filters. These are expcted to follow the following:
    extra_cui_filter ⊆ MCT filter ⊆ Model/config filter

    While any other CUIs can be included in the the extra CUI filter or
    the MCT filter, they would not have any real effect.
    """
    cuis: set[str] = set()
    cuis_exclude: set[str] = set()

    def __init__(self, **data):
        if 'cuis' in data:
            cuis = data['cuis']
            if isinstance(cuis, dict) and len(cuis) == 0:
                logger.warning("Loading an old model where "
                               "config.linking.filters.cuis has been "
                               "dict to an empty dict instead of an empty "
                               "set. Converting the dict to a set in memory "
                               "as that is what is expected. Please consider "
                               "saving the model again.")
                data['cuis'] = set(cuis.keys())
        super().__init__(**data)

    def check_filters(self, cui: str) -> bool:
        """Checks is a CUI in the filters

        Args:
            cui (str): The CUI in question

        Returns:
            bool: True if the CUI is allowed
        """
        if cui in self.cuis or not self.cuis:
            return cui not in self.cuis_exclude
        else:
            return False


class Linking(CoreComponentConfig):
    """The linking part of the config"""
    optim: dict = {'type': 'linear', 'base_lr': 1, 'min_lr': 0.00005}
    """Linear anneal"""
    # optim: dict = {'type': 'standard', 'lr': 1}
    context_vector_sizes: dict = {'xlong': 27, 'long': 18,
                                  'medium': 9, 'short': 3}
    """Context vector sizes that will be calculated and used for linking"""
    context_vector_weights: dict = {'xlong': 0.1, 'long': 0.4,
                                    'medium': 0.4, 'short': 0.1}
    """Weight of each vector in the similarity score - make trainable at
    some point. Should add up to 1."""
    filters: LinkingFilters = LinkingFilters()
    """Filters"""
    train: bool = True
    """Should it train or not, this is set automatically ignore in 99% of
    cases and do not set manually"""
    random_replacement_unsupervised: float = 0.80
    """If <1 during unsupervised training the detected term will be randomly
    replaced with a probability of 1 - random_replacement_unsupervised
    Replaced with a synonym used for that term"""
    disamb_length_limit: int = 3
    """All concepts below this will always be disambiguated"""
    filter_before_disamb: bool = False
    """If True it will filter before doing disamb. Useful for the trainer."""
    train_count_threshold: int = 1
    """Concepts that have seen less training examples than this will not be
    used for similarity calculation and will have a similarity of -1."""
    always_calculate_similarity: bool = False
    """Do we want to calculate context similarity even for concepts that are
    not ambigous."""
    calculate_dynamic_threshold: bool = False
    """Concepts below this similarity will be ignored. Type can be
    static/dynamic - if dynamic each CUI has a different TH
    and it is calcualted as the average confidence for that
    CUI * similarity_threshold. Take care that dynamic works only
    if the cdb was trained with calculate_dynamic_threshold = True."""
    similarity_threshold_type: str = 'static'
    similarity_threshold: float = 0.25
    negative_probability: float = 0.5
    """Probability for the negative context to be added for each
    positive addition"""
    negative_ignore_punct_and_num: bool = True
    """Do we ignore punct/num when negative sampling"""
    prefer_primary_name: float = 0.35
    """If >0 concepts for which a detection is its primary name
    will be preferred by that amount (0 to 1)"""
    prefer_frequent_concepts: float = 0.35
    """If >0 concepts that are more frequent will be prefered
    by a multiply of this amount"""
    subsample_after: int = 30000
    """DISABLED in code permanetly: Subsample during unsupervised
    training if a concept has received more than"""
    devalue_linked_concepts: bool = False
    """When adding a positive example, should it also be treated as Negative
    for concepts which link to the postive one via names (ambigous names)."""
    context_ignore_center_tokens: bool = False
    """If true when the context of a concept is calculated (embedding)
    the words making that concept are not taken into accout"""


class Preprocessing(BaseModel):
    """The preprocessing part of the config"""
    words_to_skip: set = {'nos'}
    """This words will be completly ignored from concepts and from the text
    (must be a Set)"""
    keep_punct: set = {'.', ':'}
    """All punct will be skipped by default, here you can set what
    will be kept"""
    do_not_normalize: set[str] = {'VBD', 'VBG', 'VBN', 'VBP', 'JJS', 'JJR'}
    """Should specific word types be normalized: e.g. running -> run
    Values are detailed part-of-speech tags. See:
    - https://spacy.io/usage/linguistic-features#pos-tagging
    - Label scheme section per model at https://spacy.io/models/en"""
    skip_stopwords: bool = False
    """Should stopwords be skipped/ingored when processing input"""
    min_len_normalize: int = 5
    """Nothing below this length will ever be normalized (input tokens or
    concept names), normalized means lemmatized in this case"""
    stopwords: Optional[set] = None
    """If None the default set of stowords from spacy will be used.
    This must be a Set.

    NB! For these changes to take effect, the pipe would need to be recreated.
    """
    max_document_length: int = 1000000
    """Documents longer  than this will be trimmed.

    NB! For these changes to take effect, the pipe would need to be recreated.
    """


class CDBMaker(BaseModel):
    """The Context Database (CDB) making part of the config"""
    name_versions: list = ['LOWER', 'CLEAN']
    """Name versions to be generated."""
    multi_separator: str = '|'
    """If multiple names or type_ids for a concept present in one row of a CSV,
    they are separted
    by the character below."""
    remove_parenthesis: int = 5
    """Should preferred names with parenthesis be cleaned 0 means no,
    else it means if longer than or equal
    e.g. Head (Body part) -> Head"""
    min_letters_required: int = 2
    """Minimum number of letters required in a name to be accepted
    for a concept"""


class Ner(CoreComponentConfig):
    """The NER part of the config"""
    min_name_len: int = 3
    """Do not detect names below this limit, skip them"""
    max_skip_tokens: int = 2
    """When checking tokens for concepts you can have skipped tokens between
    used ones (usually spaces, new lines etc). This number tells you how many
    skipped can you have."""
    check_upper_case_names: bool = False
    """Check uppercase to distinguish uppercase and lowercase words that have
    a different meaning."""
    upper_case_limit_len: int = 4
    """Any name shorter than this must be uppercase in the text to be
    considered. If it is not uppercase
    it will be skipped."""
    try_reverse_word_order: bool = False
    """Try reverse word order for short concepts (2 words max),
    e.g. heart disease -> disease heart"""


class AnnotationOutput(BaseModel):
    """The annotation output part of the config"""
    context_left: int = -1
    context_right: int = -1
    lowercase_context: bool = True
    include_text_in_output: bool = False


# NOTE: this class should have an attribute for each
#       medcat2.components.types.CoreComponentType
class Components(BaseModel):
    tagging: CoreComponentConfig = CoreComponentConfig()
    token_normalizing: CoreComponentConfig = CoreComponentConfig()
    ner: Ner = Ner()
    linking: Linking = Linking()
    comp_order: list[str] = ['tagging', 'token_normalizing',
                             'ner', 'linking']


class TrainingDescriptor(BaseModel):
    train_time: datetime = Field(default_factory=datetime.now)
    project_name: Optional[str]
    num_docs: int


class ModelMeta(BaseModel):
    description: str = 'N/A'
    ontology: list[str] = []
    hash: str = ''  # TODO - implement
    # TODO - implement
    last_saved: datetime = Field(default_factory=datetime.now)
    unsup_trained: list[TrainingDescriptor] = []  # TODO - implement
    sup_trained: list[TrainingDescriptor] = []  # TODO - implement
    saved_environ: Environment = get_environment_info()  # TODO - implement

    def mark_saved_now(self):
        self.last_saved = datetime.now()


class Config(BaseModel):
    general: General = General()
    components: Components = Components()
    # linking: Linking = Linking()
    preprocessing: Preprocessing = Preprocessing()
    cdb_maker: CDBMaker = CDBMaker()
    # ner: Ner = Ner()
    annotation_output: AnnotationOutput = AnnotationOutput()
    meta: ModelMeta = ModelMeta()
