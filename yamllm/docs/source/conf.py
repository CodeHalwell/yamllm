import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Points to root project directory

project = 'YAMLLM'
copyright = '2025, Daniel Halwell'
author = 'Daniel Halwell'

# Version info
version = '0.1.0'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'myst_parser'
]

# Theme settings
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Source file settings
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
