from typing import Iterable, Callable
import logging
from itertools import chain, repeat, islice

from medcat2.tokenizing.tokens import MutableDocument
from medcat2.cdb import CDB
from medcat2.utils.config_utils import temp_changed_config


logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, cdb: CDB, caller: Callable[[str], MutableDocument]):
        self.cdb = cdb
        self.config = cdb.config
        self.caller = caller

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
        with temp_changed_config(self.config.components.linking,
                                 'train', True):
            self._train_unsupervised(data_iterator, nepochs, fine_tune,
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
