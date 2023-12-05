import os
from itertools import chain
import importlib.util
from setuptools import find_packages, setup


spec = importlib.util.spec_from_file_location(
    "package_info", "flex_model/package_info.py"
)
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)


__contact_emails__ = package_info.__contact_emails__
__contact_names__ = package_info.__contact_names__
__description__ = package_info.__description__
__homepage__ = package_info.__homepage__
__license__ = package_info.__license__
__package_name__ = package_info.__package_name__
__repository_url__ = package_info.__repository_url__
__version__ = package_info.__version__


def req_file(filename, folder="requirements"):
    with open(os.path.join(folder, filename), encoding="utf-8") as f:
        content = f.readlines()

    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")
extras_require = {
    "core": req_file("requirements.txt"),
    "test": req_file("test_requirements.txt"),
    "dev": req_file("dev_requirements.txt"),
    "examples": req_file("examples_requirements.txt"),
}


extras_require["all"] = list(chain(extras_require.values()))
extras_require["test"] = list(
    chain([extras_require["core"], extras_require["test"]])
)
extras_require["dev"] = list(
    chain(
        [extras_require["core"], extras_require["test"], extras_require["dev"]]
    )
)
extras_require["examples"] = list(
    chain([extras_require["core"], extras_require["examples"]])
)

setup(
    name=__package_name__,
    version=__version__,
    description=__description__,
    url=__repository_url__,
    author=__contact_names__,
    author_email=__contact_emails__,
    maintainer=__contact_names__,
    maintainer_email=__contact_emails__,
    license=__license__,
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
