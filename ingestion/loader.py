from typing import Optional, Callable, List
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX

DEFAULT_BASE_URL = "https://help.mail.ru/"

def make_link_regex() -> str:
    return rf"href=[\"']?{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)[> \#][ '\"]?"

def load_documents_from_site(
    base_url: str = DEFAULT_BASE_URL,
    max_depth: int = 3,
    extractor: Optional[Callable[[str], str]] = None,
    continue_on_failure: bool = True,
) -> List:
    
    link_regex = make_link_regex()
    loader = RecursiveUrlLoader(
        base_url,
        max_depth=max_depth,
        extractor=extractor,
        continue_on_failure=continue_on_failure,
        base_url=base_url,
        link_regex=link_regex,
    )
    docs = loader.load()
    return docs
    