
import importlib.metadata

__version__ = "3.3.6"

#__version__ = importlib.metadata.version("pyLARDA")

__author__ = "pyLARDA-dev-team"
#__author__ = importlib.metadata.authors("pyLARDA")
__doc_link__ = "https://lacros-tropos.github.io/larda-doc/"

__init_text__ = f""">> LARDA initialized. Documentation available at {__doc_link__}"""

__default_info__ = """
The data from this campaign is provided by larda without warranty and liability.
Before publishing check the data license and contact the principal investigator.
Detailed information might be available using `larda.description('system', 'parameter')`."""
