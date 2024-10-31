from typing import Iterable, Callable, Optional, Protocol, Union
import logging
from itertools import chain, repeat, islice
from tqdm import trange
from contextlib import nullcontext
from datetime import datetime

from medcat2.tokenizing.tokens import (MutableDocument, MutableEntity,
                                       MutableToken)
from medcat2.cdb import CDB
from medcat2.config.config import LinkingFilters
from medcat2.utils.config_utils import temp_changed_config
from medcat2.utils.data_utils import make_mc_train_test, get_false_positives
from medcat2.utils.iterutils import callback_iterator
from medcat2.data.mctexport import (MedCATTrainerExport,
                                    MedCATTrainerExportProject)


logger = logging.getLogger(__name__)


class AdderType(Protocol):

    def __call__(selself,
                 cui: str,
                 name: str,
                 mut_doc: Optional[MutableDocument] = None,
                 mut_ent: Optional[Union[list[MutableToken],
                                         MutableEntity]] = None,
                 ontologies: set[str] = set(),
                 name_status: str = 'A',
                 type_ids: set[str] = set(),
                 description: str = '',
                 full_build: bool = True,
                 negative: bool = False,
                 devalue_others: bool = False,
                 do_add_concept: bool = True) -> None:
        pass


class Trainer:

    def __init__(self, cdb: CDB, caller: Callable[[str], MutableDocument],
                 unlinker: Callable[[str, str, bool], None],
                 adder: AdderType):
        self.cdb = cdb
        self.config = cdb.config
        self.caller = caller
        self.unlinker = unlinker
        self.adder = adder

    def train_unsupervised(self,
                           data_iterator: Iterable,
                           nepochs: int = 1,
                           fine_tune: bool = True,
                           progress_print: int = 1000,
                           #    checkpoint: Optional[Checkpoint] = None,
                           ) -> None:
        """Runs training on the data, note that the maximum length of a line
        or document is 1M characters. Anything longer will be trimmed.

        Args:
            data_iterator (Iterable):
                Simple iterator over sentences/documents, e.g. a open file
                or an array or anything that we can use in a for loop.
            nepochs (int):
                Number of epochs for which to run the training.
            fine_tune (bool):
                If False old training will be removed.
            progress_print (int):
                Print progress after N lines.
            checkpoint (Optional[medcat.utils.checkpoint.CheckpointUT]):
                The MedCAT checkpoint object
            is_resumed (bool):
                If True resume the previous training; If False, start a fresh
                new training.
        """
        with self.config.meta.prepare_and_report_training(
            data_iterator, nepochs, False
        ) as wrapped_iter:
            with temp_changed_config(self.config.components.linking,
                                     'train', True):
                self._train_unsupervised(wrapped_iter, nepochs, fine_tune,
                                         progress_print)

    def _train_unsupervised(self,
                            data_iterator: Iterable,
                            nepochs: int = 1,
                            fine_tune: bool = True,
                            progress_print: int = 1000,
                            #    checkpoint: Optional[Checkpoint] = None,
                            ) -> None:
        if not fine_tune:
            logger.info("Removing old training data!")
            self.cdb.reset_training()
        # checkpoint = self._init_ckpts(is_resumed, checkpoint)

        # latest_trained_step = checkpoint.count if checkpoint is
        # not None else 0
        latest_trained_step = 0  # TODO: add back checkpointing
        epochal_data_iterator = chain.from_iterable(repeat(data_iterator,
                                                           nepochs))
        for line in islice(epochal_data_iterator, latest_trained_step, None):
            if line is not None and line:
                # Convert to string
                line = str(line).strip()

                try:
                    _ = self.caller(line)
                except Exception as e:
                    logger.warning("LINE: '%s...' \t WAS SKIPPED", line[0:100])
                    logger.warning("BECAUSE OF: %s", str(e))
            else:
                logger.warning("EMPTY LINE WAS DETECTED AND SKIPPED")

            latest_trained_step += 1
            if latest_trained_step % progress_print == 0:
                logger.info("DONE: %s", str(latest_trained_step))
            # if (checkpoint is not None and checkpoint.steps is not None
            #         and latest_trained_step % checkpoint.steps == 0):
            #     checkpoint.save(cdb=self.cdb, count=latest_trained_step)

    def _reset_cui_counts(self, train_set: MedCATTrainerExport,
                          reset_val: int = 100):
        # Get all CUIs
        cuis = []
        for project in train_set['projects']:
            for ann in (ann for doc in project['documents']
                        for ann in doc['annotations']):
                cuis.append(ann['cui'])
        for cui in set(cuis):
            if self.cdb.cui2info[cui].count_train != 0:
                self.cdb.cui2info[cui].count_train = reset_val

    def train_supervised_raw(self,
                             data: MedCATTrainerExport,
                             reset_cui_count: bool = False,
                             nepochs: int = 1,
                             print_stats: int = 0,
                             use_filters: bool = False,
                             terminate_last: bool = False,
                             use_overlaps: bool = False,
                             use_cui_doc_limit: bool = False,
                             test_size: float = 0,
                             devalue_others: bool = False,
                             use_groups: bool = False,
                             never_terminate: bool = False,
                             train_from_false_positives: bool = False,
                             extra_cui_filter: Optional[set[str]] = None,
                             #  checkpoint: Optional[Checkpoint] = None,
                             disable_progress: bool = False,
                             ) -> tuple:
        """Train supervised based on the raw data provided.

        The raw data is expected in the following format:
        {'projects':
            [ # list of projects
                { # project 1
                    'name': '<some name>',
                    # list of documents
                    'documents': [{'name': '<some name>',  # document 1
                                    'text': '<text of the document>',
                                    # list of annotations
                                    'annotations': [# annotation 1
                                                    {'start': -1,
                                                    'end': 1,
                                                    'cui': 'cui',
                                                    'value': '<text value>'},
                                                    ...],
                                    }, ...]
                }, ...
            ]
        }

        Please take care that this is more a simulated online training then
        upervised.

        When filtering, the filters within the CAT model are used first,
        then the ones from MedCATtrainer (MCT) export filters,
        and finally the extra_cui_filter (if set).
        That is to say, the expectation is:
        extra_cui_filter ⊆ MCT filter ⊆ Model/config filter.

        Args:
            data (Dict[str, List[Dict[str, dict]]]):
                The raw data, e.g from MedCATtrainer on export.
            reset_cui_count (bool):
                Used for training with weight_decay (annealing). Each concept
                has a count that is there from the beginning of the CDB, that
                count is used for annealing. Resetting the count will
                significantly increase the training impact. This will reset
                the count only for concepts that exist in the the training
                data.
            nepochs (int):
                Number of epochs for which to run the training.
            print_stats (int):
                If > 0 it will print stats every print_stats epochs.
            use_filters (bool):
                Each project in medcattrainer can have filters, do we want to
                respect those filters
                when calculating metrics.
            terminate_last (bool):
                If true, concept termination will be done after all training.
            use_overlaps (bool):
                Allow overlapping entities, nearly always False as it is very
                difficult to annotate overlapping entities.
            use_cui_doc_limit (bool):
                If True the metrics for a CUI will be only calculated if that
                CUI appears in a document, in other words if the document was
                annotated for that CUI. Useful in very specific situations
                when during the annotation process the set of CUIs changed.
            test_size (float):
                If > 0 the data set will be split into train test based on
                this ration. Should be between 0 and 1. Usually 0.1 is fine.
            devalue_others(bool):
                Check add_name for more details.
            use_groups (bool):
                If True concepts that have groups will be combined and stats
                will be reported on groups.
            never_terminate (bool):
                If True no termination will be applied
            train_from_false_positives (bool):
                If True it will use false positive examples detected by medcat
                and train from them as negative examples.
            extra_cui_filter(Optional[Set]):
                This filter will be intersected with all other filters, or if
                all others are not set then only this one will be used.
            checkpoint (Optional[Optional[medcat.utils.checkpoint.Checkpoint]):
                The MedCAT Checkpoint object
            disable_progress (bool):
                Whether to disable the progress output (tqdm). Defaults to
                False.

        Returns:
            Tuple: Consisting of the following parts
                fp (dict):
                    False positives for each CUI.
                fn (dict):
                    False negatives for each CUI.
                tp (dict):
                    True positives for each CUI.
                p (dict):
                    Precision for each CUI.
                r (dict):
                    Recall for each CUI.
                f1 (dict):
                    F1 for each CUI.
                cui_counts (dict):
                    Number of occurrence for each CUI.
                examples (dict):
                    FP/FN examples of sentences for each CUI.
        """
        # checkpoint = self._init_ckpts(is_resumed, checkpoint)

        # the config.linking.filters stuff is used directly in
# medcat.linking.context_based_linker and medcat.linking.vector_context_model
        # as such, they need to be kept up to date with per-project filters
        # However, the original state needs to be kept track of
        # so that it can be restored after training

        fp: dict[str, int] = {}
        fn: dict[str, int] = {}
        tp: dict[str, int] = {}
        p: dict[str, float] = {}
        r: dict[str, float] = {}
        f1: dict[str, float] = {}
        examples: dict[str, object] = {}

        cui_counts: dict[str, int] = {}

        if test_size == 0:
            logger.info("Running without a test set, or train==test")
            test_set = data
            train_set = data
        else:
            train_set, test_set, _, _ = make_mc_train_test(data, self.cdb,
                                                           test_size=test_size)

    # if print_stats > 0:
    #     fp, fn, tp, p, r, f1, cui_counts, examples = self._print_stats(
    #         test_set, use_project_filters=use_filters,
    #         use_cui_doc_limit=use_cui_doc_limit, use_overlaps=use_overlaps,
    #         use_groups=use_groups, extra_cui_filter=extra_cui_filter)
        if reset_cui_count:
            self._reset_cui_counts(train_set)

        # Remove entities that were terminated
        if not never_terminate:
            for ann in (ann for project in train_set['projects']
                        for doc in project['documents']
                        for ann in doc['annotations']):
                if ann.get('killed', False):
                    # self.unlink_concept_name(ann['cui'], ann['value'])
                    self.unlinker(ann['cui'], ann['value'], False)

        # latest_trained_step = (checkpoint.count if checkpoint is not None
        #                        else 0)
        # (current_epoch,
        #  current_project,
        #  current_document) = self._get_training_start(train_set,
        #                                               latest_trained_step)
        current_epoch = 0
        current_project = 0
        current_document = 0

        for epoch in trange(current_epoch, nepochs, initial=current_epoch,
                            total=nepochs, desc='Epoch', leave=False,
                            disable=disable_progress):
            self._perform_epoch(current_project, current_document, train_set,
                                disable_progress, extra_cui_filter,
                                use_filters, train_from_false_positives,
                                devalue_others, terminate_last,
                                never_terminate)

        # if print_stats > 0 and (epoch + 1) % print_stats == 0:
        #     fp, fn, tp, p, r, f1, cui_counts, examples = self._print_stats(
        #         test_set, epoch=epoch + 1, use_project_filters=use_filters,
        #         use_cui_doc_limit=use_cui_doc_limit,
        #         use_overlaps=use_overlaps, use_groups=use_groups,
        #         extra_cui_filter=extra_cui_filter)

        # # reset the state of filters
        # self.config.linking.filters = orig_filters

        return fp, fn, tp, p, r, f1, cui_counts, examples

    def _perform_epoch(self, current_project: int,
                       current_document: int,
                       train_set: MedCATTrainerExport,
                       disable_progress: bool,
                       extra_cui_filter: Optional[set[str]],
                       use_filters: bool,
                       train_from_false_positives: bool,
                       devalue_others: bool,
                       terminate_last: bool,
                       never_terminate: bool,
                       ) -> None:
        # Print acc before training
        for idx_project in trange(current_project,
                                  len(train_set['projects']),
                                  initial=current_project,
                                  total=len(train_set['projects']),
                                  desc='Project', leave=False,
                                  disable=disable_progress):
            project = train_set['projects'][idx_project]
            with self._project_filters(
                    self.config.components.linking.filters, project,
                    extra_cui_filter, use_filters):
                self._train_supervised_for_project(
                    project, current_document, train_from_false_positives,
                    devalue_others)

        if terminate_last and not never_terminate:
            # Remove entities that were terminated,
            # but after all training is done
            # for project in train_set['projects']:
            #     for doc in project['documents']:
            #         for ann in doc_annotations:
            for ann in (ann for project in train_set['projects']
                        for doc in project['documents']
                        for ann in doc['annotations']):
                if ann.get('killed', False):
                    # # TODO: unlink
                    # self.unlink_concept_name(ann['cui'], ann['value'])
                    self.unlinker(ann['cui'], ann['value'], False)

    def _project_filters(self, filters: LinkingFilters,
                         project: MedCATTrainerExportProject,
                         extra_cui_filter: Optional[set[str]],
                         use_project_filters: bool):
        if extra_cui_filter is not None and not use_project_filters:
            return temp_changed_config(filters, 'cuis', extra_cui_filter)
        if use_project_filters:
            cuis = project.get('cuis', None)
            if cuis is None or not cuis:
                return nullcontext()
            return temp_changed_config(filters, 'cuis', cuis)
        return temp_changed_config(filters, 'cuis', set())

    def _train_supervised_for_project(self,
                                      project: MedCATTrainerExportProject,
                                      current_document: int,
                                      train_from_false_positives: bool,
                                      devalue_others: bool):
        cnf_linking = self.config.components.linking
        for idx_doc in trange(current_document,
                              len(project['documents']),
                              initial=current_document,
                              total=len(project['documents']),
                              desc='Document', leave=False):
            doc = project['documents'][idx_doc]
            mut_doc = self.caller(doc['text'])  # type: ignore

            # Compatibility with old output where annotations are a list
            for ann in doc['annotations']:
                if not ann.get('killed', False):
                    cui = ann['cui']
                    start = ann['start']
                    end = ann['end']
                    mut_entity = mut_doc[start: end]
                    deleted = ann.get('deleted', False)
                    if cnf_linking.filters.check_filters(cui):
                        # TODO: allow adding/training
                        self.adder(
                            cui=cui, name=ann['value'], mut_doc=mut_doc,
                            mut_ent=mut_entity, negative=deleted,
                            devalue_others=devalue_others)
            if train_from_false_positives:
                fps: list[MutableEntity] = get_false_positives(doc, mut_doc)

                for fp in fps:  # type: ignore
                    fp_: MutableEntity = fp  # type: ignore
                    # TODO: allow adding/training
                    self.adder(cui=fp_.cui, name=fp_.base.text,
                               mut_doc=mut_doc, mut_ent=fp_,
                               negative=True, do_add_concept=False)

            # latest_trained_step += 1
            # if (checkpoint is not None and checkpoint.steps is not None
            #         and latest_trained_step % checkpoint.steps == 0):
            #     checkpoint.save(self.cdb, latest_trained_step)
