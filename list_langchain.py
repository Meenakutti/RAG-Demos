import pkgutil
import langchain
mods=[m.name for m in pkgutil.walk_packages(langchain.__path__, langchain.__name__+'.')]
print([m for m in mods if 'chunk' in m.lower()])
