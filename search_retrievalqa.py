import pkgutil, importlib, langchain
found=False
for finder, name, ispkg in pkgutil.walk_packages(langchain.__path__, langchain.__name__+'.'):
    try:
        mod = importlib.import_module(name)
        if hasattr(mod, 'RetrievalQA'):
            print('module', name)
            found=True
    except Exception:
        pass
if not found:
    print('not found in langchain package')
