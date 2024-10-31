import pkg_resources
import platform
import logging
import importlib.metadata

from pydantic import BaseModel


logger = logging.getLogger(__name__)


def get_direct_dependencies() -> list[str]:
    """Gets the direct dependencies of the current package and their versions.
    """
    # NOTE: __package__ would be medcat2.utils in this case
    package = __package__.split('.', 1)[0]
    reqs = importlib.metadata.requires(package)
    if reqs is None:
        raise ValueError("Unable to find package direct dependencies")
    return reqs


def _update_installed_dependencies_recursive(
        gathered: dict[str, str],
        package: pkg_resources.Distribution) -> dict[str, str]:
    if package.project_name in gathered:
        logger.debug("Trying to update already found transitive dependency "
                     "'%'", package.egg_name)
        return gathered
    for req in package.requires():
        if req.project_name in gathered:
            logger.debug("Trying to look up already found transitive "
                         "dependency '%'", req.project_name)
            continue  # don't look for it again
        try:
            dep = pkg_resources.get_distribution(req.project_name)
        except pkg_resources.DistributionNotFound as e:
            logger.warning("Unable to locate requirement '%s':",
                           req.project_name, exc_info=e)
            continue
        gathered[dep.project_name] = dep.version
        _update_installed_dependencies_recursive(gathered, dep)
    return gathered


def get_transitive_deps(direct_deps: list[str]) -> dict[str, str]:
    """Get the transitive dependencies of the direct dependencies.

    Args:
        direct_deps (List[str]): List of direct dependencies.

    Returns:
        dict[str, str]: The dependency names and their corresponding versions.
    """
    # map from name to version so as to avoid multiples of the same package
    all_transitive_deps: dict[str, str] = {}
    for dep in direct_deps:
        package = pkg_resources.get_distribution(dep)
        _update_installed_dependencies_recursive(all_transitive_deps, package)
    return all_transitive_deps


def get_installed_packages() -> dict[str, str]:
    """Get the installed packages and their versions.

    Returns:
        dict[str, str]: All installed packages and their versions.
    """
    direct_deps = get_direct_dependencies()
    installed_packages: dict[str, str] = {}
    for package in pkg_resources.working_set:
        if package.project_name not in direct_deps:
            continue
        installed_packages[package.project_name] = package.version
    return installed_packages


class Environment(BaseModel):
    dependencies: dict[str, str]
    transitive_deps: dict[str, str]
    os: str
    cpu_arcitecture: str
    python_version: str


def get_environment_info(include_transitive_deps: bool = True) -> Environment:
    """Get the current environment information.

    This includes dependency versions, the OS, the CPU architecture and the
        python version.

    Args:
        include_transitive_deps (bool): Whether to include transitive
            dependencies. Defaults to True.

    Returns:
        Dict[str, Any]: _description_
    """
    deps = get_installed_packages()
    os = platform.platform()
    cpu_arc = platform.machine()
    py_ver = platform.python_version()
    if include_transitive_deps:
        direct_deps = list(deps.keys())
        trans_deps = get_transitive_deps(direct_deps)
    else:
        trans_deps = {}
    return Environment(dependencies=deps, transitive_deps=trans_deps, os=os,
                       cpu_arcitecture=cpu_arc, python_version=py_ver)
