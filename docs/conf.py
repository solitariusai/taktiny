# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add src/ to sys.path so autodoc can find taktiny
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import taktiny

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Taktiny"
copyright = "2026, Shinapri"
author = "Shinapri"
release = getattr(taktiny, "__version__", "0.0.1")
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

# Copybutton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Autodoc settings
autodoc_default_options = {
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Napoleon settings for Google/NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# MyST parser extensions
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "attrs_inline",
    "attrs_block",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "Taktiny"
html_logo = "_static/mark.svg"
html_favicon = "_static/favicon.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["theme.js"]
pygments_style = "bw"
pygments_dark_style = "bw"

html_theme_options = {
    "source_repository": "https://github.com/solitariusai/taktiny",
    "source_branch": "experiment",
    "source_directory": "docs/",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_css_variables": {
        # The Pygments bw style has a white background in both modes. Override
        # Furo's derived code colors explicitly, including its auto-mode CSS.
        "color-code-background": "#ffffff",
        "color-code-foreground": "#111111",
        "color-brand-primary": "#111111",
        "color-brand-content": "#111111",
        "color-foreground-primary": "#111111",
        "color-foreground-secondary": "#333333",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#ffffff",
        "color-background-hover": "#eeeeee",
        "color-background-border": "#111111",
        "color-admonition-background": "#ffffff",
        "font-stack": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
    },
    "dark_css_variables": {
        "color-code-background": "#111111",
        "color-code-foreground": "#ffffff",
        "color-brand-primary": "#ffffff",
        "color-brand-content": "#ffffff",
        "color-foreground-primary": "#ffffff",
        "color-foreground-secondary": "#dddddd",
        "color-background-primary": "#111111",
        "color-background-secondary": "#111111",
        "color-background-hover": "#292929",
        "color-background-border": "#ffffff",
        "color-admonition-background": "#111111",
        "font-stack": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/solitariusai/taktiny",
            "html": """<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>""",  # noqa: E501
            "class": "",
        },
    ],
}
