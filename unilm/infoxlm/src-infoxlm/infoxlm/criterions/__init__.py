import os
import importlib


# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
  if file.endswith('.py') and not file.startswith('_'):
    module = file[:file.find('.py')]
    importlib.import_module('infoxlm.criterions.' + module)