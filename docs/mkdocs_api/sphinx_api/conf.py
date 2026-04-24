
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'CineInfini'
copyright = '2026, Salah-Eddine BENBRAHIM'
author = 'Salah-Eddine BENBRAHIM'
release = '0.1.2'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Éviter le dossier _modules en mettant tout à la racine
# Pas de répertoire séparé pour les sources
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_extra_path = []  # Éviter les fichiers supplémentaires

# Options pour autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
