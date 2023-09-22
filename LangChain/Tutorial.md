
```
from langchain.llms import LlamaCpp 
from langchain import PromptTemplate, LLMChain 
from langchain.callbacks.manager import CallbackManager 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
```

```
template = """Question: {question} Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
```

```
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# Make sure the model path is correct for your system!
llm = LlamaCpp( model_path="./ggml-model-q4_0.bin", callback_manager=callback_manager, verbose=True )
```

```
llm_chain = LLMChain(prompt=prompt, llm=llm)
```

```
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?" 

llm_chain.run(question)
```

> When was Justin Bieber born? He was born on March 1, 1994. The Super Bowl is usually played in February. So, the Super Bowl that took place in the year Justin Bieber was born would be Super Bowl XLVII, which was held on February 3, 2013. The NFL team that won that Super Bowl was the Baltimore Ravens

