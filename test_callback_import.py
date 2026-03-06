try:
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    print('import succeeded from langchain.callbacks')
except Exception as e:
    print('failed langchain.callbacks', e)

try:
    from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    print('import succeeded from langchain_core.callbacks')
except Exception as e:
    print('failed langchain_core.callbacks', e)
