# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import configparser
import datetime
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'Geomapi'
#copyright = '2022, Bassier, De Geyter, De Winter, Vermandere'
#author = 'Bassier, De Geyter, De Winter, Vermandere'

# The full version, including alpha/beta/rc tags
#release = '0.0.8'
config = configparser.ConfigParser()
config.read("../../setup.cfg")
release = config["metadata"]["version"]
author = config["metadata"]["author"]
copyright = str(datetime.date.today().year) + ", " + author


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    #'myst_parser',
    'myst_nb'
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True
autodoc_mock_imports = [
    "open3d", 
    "opencv-python", 
    "pye57",  
    "rdflib", 
    "cv2", 
    "typing_extensions", 
    "matplotlib",
    "ifcopenshell",
    "scipy",
    "PIL",
    "xlsxwriter",
    "trimesh",
    "mpl_toolkits",
    "fcl",
    "osgeo",
    "sklearn",
    "ezdxf",
    "numpy-quaternion",
    "pandas",
    "laspy"
    ]
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Options for the notebook parsing
nb_execution_mode = "off"

# Options for Myst markdown parsing
myst_enable_extensions = ["dollarmath", "amsmath", "html_image"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Sorting the functions by name
autodoc_member_order = 'alphabetical'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "_static/geomapi_logo.png"
html_favicon = '_static/favicon.ico'
html_theme_options = {
    'logo_only': True,
}

# initial Build instructions:
# cd ./docs 
# sphinx-quickstart 
# sphinx-apidoc -o . ../geomapi/
# ./make html

# 
# sphinx-apidoc -o . ../geomapi/
# sphinx-build -b html docs/source/ docs/_build
