from typing import Optional, Set
from multiprocessing import cpu_count


DEFAULT_SPACY_MODEL = 'en_core_web_md'


# defaults for config
def weighted_average(step: int, factor: float) -> float:
    return max(0.1, 1 - (step ** 2 * factor))


def default_weighted_average(step: int) -> float:
    return weighted_average(step, factor=0.0004)


def workers(workers_override: Optional[int] = None) -> int:
    if workers_override is not None:
        return workers_override
    return max(cpu_count() - 1, 1)


class StatusTypes:
    PRIMARY_STATUS_NO_DISAMB = 'P'
    PRIMARY_STATUS_W_DISAMB = 'PD'
    PRIMARY_STATUS: Set[str] = {PRIMARY_STATUS_NO_DISAMB,
                                PRIMARY_STATUS_W_DISAMB}
    MUST_DISAMBIGATE = 'N'
    AUTOMATIC = 'A'
    ALLOWED_STATUS = {PRIMARY_STATUS_NO_DISAMB, MUST_DISAMBIGATE, AUTOMATIC}
    DO_DISAMBUGATION = {MUST_DISAMBIGATE, PRIMARY_STATUS_W_DISAMB}
