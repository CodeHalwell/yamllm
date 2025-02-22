import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'YAMLLM'
copyright = '2025, Daniel Halwell'
author = 'Daniel Halwell'

version = '0.1.5'
release = '0.1.5'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'autoapi.extension'
]

# Remove viewcode temporarily due to parsing issues
# 'sphinx.ext.viewcode' removed

html_theme = 'sphinx_rtd_theme'

# AutoAPI settings
autoapi_type = 'python'
autoapi_dirs = ['../../yamllm']
autoapi_template_dir = '_templates'
autoapi_file_patterns = ['*.py']
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]
autoapi_add_toctree_entry = True
autoapi_keep_files = True
autoapi_python_class_content = 'both'
autoapi_python_use_implicit_namespaces = True
autoapi_generate_api_docs = True
autoapi_member_order = 'groupwise'

# Intersphinx mappings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# MyST parser settings
myst_enable_extensions = [
    'colon_fence'
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
