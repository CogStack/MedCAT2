import logging


logger = logging.getLogger(__name__)


def has_spacy_model(model_name: str) -> bool:
    """Checks if the spacy model is available.

    Args:
        model_name (str): The model name.

    Returns:
        bool: True if the model is available, False otherwise.
    """
    import spacy.util
    return model_name in spacy.util.get_installed_models()


def ensure_spacy_model(model_name: str) -> None:
    """Ensure the specified spacy model exists.

    If the model does not currently exist, it will attempt downloading it.

    Args:
        model_name (str): The spacy model name.
    """
    if has_spacy_model(model_name):
        return
    import subprocess
    # running in subprocess so that we can catch the exception
    # if the model name is unknown. Otherwise we'd just be bumped
    # out of python (sys.exit).
    logger.info("Installing the spacy model %s using the CLI command "
                "'python -m spacy download %s'", model_name, model_name)
    subprocess.run(["python", "-m", "spacy", "download", model_name],
                   check=True)
