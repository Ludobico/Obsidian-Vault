- [[#RunnableConfig 가 필요한 이유|RunnableConfig 가 필요한 이유]]
- [[#configurable|configurable]]
- [[#왜 Configurable을 쓰는 것이 좋은가|왜 Configurable을 쓰는 것이 좋은가]]


`RunnableConfig`  는 [[LangChain/LangChain]] 에서 [[Runnable]] 을 실행할때 함께 전달되는 config 정보 입니다.

Runnable이 무엇을 할지 정의한다면, `RunnableConfig` 는 **그 일을 어떤 조건에서 수행**할지를 정합니다.

langchain 에서는 [[prompts]] , [[output_parsers]] , [[chains]] 등 거의 모든 실행 단위가 Runnable 로 표현됩니다.
이떄 각 Runnable은 입력을 받아 출력을 만드는 로직에만 집중하고,
실행과 관련된 부가 정보는 <font color="#ffff00">RunnableConfig</font> 를 통해 외부에서 전달받도록 설계되어 있습니다.

## RunnableConfig 가 필요한 이유

일반적인 [[Python]] 코드에서는 함수 실행 시점에 다음과 같은 정보를 함께 다루기 어렵습니다.

- 이 실행이 어떤 목적을 가지는지
- 실행 과정을 어떻게 기록할지
- 동시에 몇 개까지 실행할 수 있는지
- 실행 중에 사용할 LLM이나 외부 객체는 무엇인지

이런 정보들을 모두 함수 인자로 넘기기 시작하면, 코드는 금방 복잡해지고 재사용하기 어려워집니다.

LangChain은 이런 문제를 해결하기 위해, **실행에 필요한 설정들을 하나의 객체로 모아서 전달하는 방식**을 선택했습니다.

이 역할을 하는 것이 `RunnableConfig` 입니다.

RunnableConfig의 중요한 특징 중 하나는, 한 번 전달되면 **체인 내부의 모든 단계로 자동으로 전달된다는 점**입니다.

```python
prompt | llm | parser
```

이 체인을 실행할 때 RunnableConfig를 함께 넘기면 prompt, llm, parser 모두 동일한 설정을 공유하게 됩니다.

그래서 태그, 콜백, LLM 설정 등을 한 곳에서만 지정해도 전체 실행에 일관되게 적용할 수 있습니다.

## configurable

`configurable` 은 RunnableConfig 에서 **가장 자주 사용되는 영역**입니다. 실무에서는 거의 항상 이 항목을 사용하게 됩니다.

`configurable` 은 **실행 중에 사용할 외부 객체를 전달하는 공간** 입니다. 대표적으로 LLM, Retriever, Tool 같은 객체들이 여기에 들어갑니다.

```python
config = {
    "configurable": {
        "llm": selected_llm
    }
}
```

```python
llm: LLM = config["configurable"]["llm"]
```

## 왜 Configurable을 쓰는 것이 좋은가

- 전역 변수에 의존하지 않아도 됩니다.
- Runnable 코드를 수정하지 않고도 LLM을 교체할 수 있습니다.
- 테스트, 실험, 운영 환경을 쉽게 나눌 수 있습니다.

