- [[#Installation|Installation]]
- [[#Types|Types]]
	- [[#Types#**PropTypes.string**|**PropTypes.string**]]
	- [[#Types#**PropTypes.number**|**PropTypes.number**]]
	- [[#Types#**PropTypes.boolean**|**PropTypes.boolean**]]
	- [[#Types#**PropTypes.object**|**PropTypes.object**]]
	- [[#Types#**PropTypes.array**|**PropTypes.array**]]
	- [[#Types#**PropTypes.func**|**PropTypes.func**]]
	- [[#Types#**PropTypes.symbol**|**PropTypes.symbol**]]
	- [[#Types#**PropTypes.instanceOf(Constructor)**|**PropTypes.instanceOf(Constructor)**]]
	- [[#Types#**PropTypes.oneOf([val1, val2, ...])**|**PropTypes.oneOf([val1, val2, ...])**]]
	- [[#Types#**PropTypes.oneOfType([type1, type2, ...])**|**PropTypes.oneOfType([type1, type2, ...])**]]
	- [[#Types#**PropTypes.arrayOf(type)**|**PropTypes.arrayOf(type)**]]
	- [[#Types#**PropTypes.objectOf(type)**|**PropTypes.objectOf(type)**]]
	- [[#Types#**PropTypes.shape({ key: PropTypes.type })**|**PropTypes.shape({ key: PropTypes.type })**]]


React 및 React Native에서 prop-types 패키지는 <font color="#ffff00">컴포넌트의 속성(props)에 대한 유형 검사를 수행하는 도구</font>입니다. 이를 통해 개발자는 컴포넌트가 올바른 props를 받았는지를 확인할 수 있습니다. 이는 개발자의 실수로 인한 오류를 줄이고, 코드의 안정성을 향상시키는 데 도움이 됩니다.

prop-types 를 사용하여 컴포넌트의 props에 대한 기대되는 유형을 정의할 수 있습니다. 예를 들어, 문자열, 숫자, 불리언, 객체, 배열 등과 같은 다양한 유형을 검사할 수 있습니다.

## Installation
---
prop-types를 사용하기 위해선 npm 또는 yarn을 통해 별도의 패키지 설치가 필요합니다.

```
yarn add prop-types
```


## Types
---

### **PropTypes.string**

주어진 속성이 문자열인지를 검증합니다.

```js
import PropTypes from 'prop-types';

function MyComponent(props) {
  return <div>{props.name}</div>;
}

MyComponent.propTypes = {
  name: PropTypes.string
};

// Usage:
<MyComponent name="John" />; // 올바른 사용
<MyComponent name={42} />;   // 올바르지 않은 사용: 문자열이 아닌 숫자가 전달됨

```

### **PropTypes.number**

주어진 속성이 숫자인지를 검증합니다.

```js
import PropTypes from 'prop-types';

function MyComponent(props) {
  return <div>{props.age}</div>;
}

MyComponent.propTypes = {
  age: PropTypes.number
};

// Usage:
<MyComponent age={25} />; // 올바른 사용
<MyComponent age="25" />;  // 올바르지 않은 사용: 문자열이 전달됨

```

### **PropTypes.boolean**

주어진 속성이 bool 값인지를 검증합니다.

```js
import PropTypes from 'prop-types';

function MyComponent(props) {
  return <div>{props.isActive ? 'Active' : 'Inactive'}</div>;
}

MyComponent.propTypes = {
  isActive: PropTypes.bool
};

// Usage:
<MyComponent isActive={true} />;  // 올바른 사용
<MyComponent isActive={false} />; // 올바른 사용
<MyComponent isActive="true" />;  // 올바르지 않은 사용: 문자열이 전달됨

```

### **PropTypes.object**

주어진 속성이 객체인지를 검증합니다.

```js
import PropTypes from 'prop-types';

function MyComponent(props) {
  return <div>{props.user.name}</div>;
}

MyComponent.propTypes = {
  user: PropTypes.object
};

// Usage:
<MyComponent user={{ name: 'John' }} />; // 올바른 사용
<MyComponent user="John" />;            // 올바르지 않은 사용: 객체가 아닌 문자열이 전달됨

```

### **PropTypes.array**

주어진 속성이 배열인지를 검증합니다.

```js
import PropTypes from 'prop-types';

function MyComponent(props) {
  return <div>{props.items.join(', ')}</div>;
}

MyComponent.propTypes = {
  items: PropTypes.array
};

// Usage:
<MyComponent items={['apple', 'banana', 'orange']} />; // 올바른 사용
<MyComponent items="apple" />;                         // 올바르지 않은 사용: 배열이 아닌 문자열이 전달됨

```

### **PropTypes.func**

주어진 속성이 함수인지를 검증합니다.

```js
import PropTypes from 'prop-types';

function MyComponent(props) {
  return <button onClick={props.onClick}>Click Me</button>;
}

MyComponent.propTypes = {
  onClick: PropTypes.func
};

// Usage:
<MyComponent onClick={() => console.log('Button clicked')} />; // 올바른 사용
<MyComponent onClick="handleClick" />;                        // 올바르지 않은 사용: 함수가 아닌 문자열이 전달됨

```

### **PropTypes.symbol**

주어진 속성이 심볼인지를 검증합니다.

```js
import PropTypes from 'prop-types';

const mySymbol = Symbol('mySymbol');

function MyComponent(props) {
  return <div>{String(props.symbol)}</div>;
}

MyComponent.propTypes = {
  symbol: PropTypes.symbol
};

// Usage:
<MyComponent symbol={mySymbol} />; // 올바른 사용
<MyComponent symbol="mySymbol" />; // 올바르지 않은 사용: 심볼이 아닌 문자열이 전

```

### **PropTypes.instanceOf(Constructor)**

주어진 속성이 특정 클래스의 인스턴스인지를 검증합니다.

```js
import PropTypes from 'prop-types';

class Person {
  constructor(name) {
    this.name = name;
  }
}

function Greeting(props) {
  return <div>Hello, {props.person.name}</div>;
}

Greeting.propTypes = {
  person: PropTypes.instanceOf(Person)
};

// Usage:
const john = new Person('John');
<Greeting person={john} />; // 올바른 사용
<Greeting person={{ name: 'John' }} />; // 올바르지 않은 사용: Person 클래스의 인스턴스가 아닌 객체가 전달됨

```

### **PropTypes.oneOf([val1, val2, ...])**

주어진 값들 중 하나인지를 검증합니다.

```js
import PropTypes from 'prop-types';

function Button(props) {
  return <button>{props.variant}</button>;
}

Button.propTypes = {
  variant: PropTypes.oneOf(['primary', 'secondary', 'tertiary'])
};

// Usage:
<Button variant="primary" />;   // 올바른 사용
<Button variant="secondary" />; // 올바른 사용
<Button variant="warning" />;   // 올바르지 않은 사용: 주어진 값 중에 없는 'warning' 전달됨

```

### **PropTypes.oneOfType([type1, type2, ...])**

주어진 여러 타입 중 하나인지를 검증합니다.

```js
import PropTypes from 'prop-types';

function MyComponent(props) {
  return <div>{props.value}</div>;
}

MyComponent.propTypes = {
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number])
};

// Usage:
<MyComponent value="Hello" />; // 올바른 사용
<MyComponent value={42} />;    // 올바른 사용
<MyComponent value={true} />;  // 올바르지 않은 사용: 문자열 또는 숫자가 아닌 부울 전달됨

```

### **PropTypes.arrayOf(type)**

주어진 배열의 모든 요소가 특정 타입인지를 검증합니다.

```js
import PropTypes from 'prop-types';

function MyComponent(props) {
  return <ul>{props.items.map(item => <li key={item}>{item}</li>)}</ul>;
}

MyComponent.propTypes = {
  items: PropTypes.arrayOf(PropTypes.string)
};

// Usage:
<MyComponent items={['apple', 'banana', 'orange']} />; // 올바른 사용
<MyComponent items={['apple', 42, 'orange']} />;       // 올바르지 않은 사용: 숫자가 포함된 배열 전달됨

```

### **PropTypes.objectOf(type)**

주어진 객체의 모든 값이 특정 타입인지를 검증합니다.

```js
import PropTypes from 'prop-types';

function MyComponent(props) {
  return <ul>{Object.values(props.data).map(value => <li key={value}>{value}</li>)}</ul>;
}

MyComponent.propTypes = {
  data: PropTypes.objectOf(PropTypes.number)
};

// Usage:
<MyComponent data={{ apple: 5, banana: 10, orange: 7 }} />; // 올바른 사용
<MyComponent data={{ apple: 'five', banana: 10, orange: 7 }} />; // 올바르지 않은 사용: 문자열이 포함된 객체 전달됨

```

### **PropTypes.shape({ key: PropTypes.type })**

주어진 객체의 각 키가 특정 타입인지를 검증합니다.

```js
import PropTypes from 'prop-types';

function MyComponent(props) {
  return <div>{props.user.name} ({props.user.age} years old)</div>;
}

MyComponent.propTypes = {
  user: PropTypes.shape({
    name: PropTypes.string,
    age: PropTypes.number
  })
};

// Usage:
<MyComponent user={{ name: 'John', age: 30 }} />; // 올바른 사용
<MyComponent user={{ name: 'John' }} />;         // 올바르지

```
