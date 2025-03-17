# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ScaleDP"
copyright = "2024, StabRise"
author = "StabRise"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "myst_parser"]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_favicon = "_static/favicon.ico"
html_logo = "https://raw.githubusercontent.com/StabRise/ScaleDP/refs/heads/master/images/scaledp.webp"

html_sidebars = {"reference/blog/*": ["navbar-logo.html", "search-field.html"]}

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/StabRise/scaledp/",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/scalepd/",
            "icon": "https://img.shields.io/pypi/v/scaledp.svg",
            "type": "url",
        },
        {
            "name": "by StabRise",
            "url": "https://stabrise.com",
            "icon": "https://img.shields.io/badge/by-StabRise-orange.svg?style=flat&colorA=E1523D&colorB=007D8A",
            "type": "url",
        },
    ]
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
