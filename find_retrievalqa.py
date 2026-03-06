import pkgutil, inspect
import langchain

for finder, name, ispkg in pkgutil.walk_packages(langchain.__path__, langchain.__name__ + '.'):
    if 'RetrievalQA' in name or 'retrievalqa' in name.lower():
        print('module', name)
        try:
            mod = __import__(name, fromlist=['*'])
            if hasattr(mod, 'RetrievalQA'):
                print('found class in', name)
        except Exception as e:
            pass
