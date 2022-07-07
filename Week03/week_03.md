
### MLE, MAP
- MLE와 MAP는 확률모델을 이용하여 어떤 변수(Θ)를 추정하는 방법
- Maximum Likelihood Estimation(MLE)
  - Likelihood probability(우도) P(X|Θ)가 최대가 되는 어떤 변수(Θ)를 찾는 방법
- Maximum A posterior Probability(MAP)
  - A posterior probability(사후확률) P(Θ|X)가 최대가 되는 어떤 변수(Θ)를 찾는 방법

### Optimal Classification
- x 라는 값이 관측 되면 y = 0 일 확률과 y = 1 일 확률을 계산하는 과정 
- $f^*$ = $argmin_{f}P(f(x) \neq  Y)$
- $f^{*}(x)$ = $argmax_{Y=y}P(Y=y | X =x)$

### Optimal Classification and Bayes Risk
- classification error를 줄일 수 있는 곡선을 구해야함
  - Decision Boundary 곡선 외 부분 

### Learning the Optimal Classifier
- Optimal Classifier 
  - Class Conditional Density, Class Prior로 이루어짐 

### Naive Bayes Classifier 
- Conditional Independence를 도입해야함 
- 모든 x는 독립이라고 가정, $P(X, Y) = P(X)P(Y)$
  - 독립적이지 않다면 $(2^d-1)k$개의 많은 파라미터를 계산해야함 
  - 독립적이라면 $(2-1)dk$개의 파라미터들을 가짐 
- 즉, 개별 input feature인 x들은 독립적이라고 가정하고 classification을 수행하는 것 
  - 나이브하게 계산 하는 것
  
### Conditional vs. Marginal Independence
- x가 다른 x에 의해 영향을 받는다면 Marginal Independence
- x가 다른 x와는 상관 없이 결과 값을 가진다면 Conditional Independence 인 것
- conditional independent assumption을 적용한다면 개별 feature들의 곱셈으로 연산이 변형됨

### Naive Bayes Classifier
- assumption is naive
- d개의 conditionally independent feature x 데이터세트가 있을 때 likelihood를 계산 할 수 있음

### Problem of Naive Bayes Classifier
- Naive assumption
  - 현실적이지 않음
- Incorrect Probability Esitmations
  - MLE는 관측되지 않은 정보라면 제대로 동작하지 않기 때문에 MAP로 다루어야함 
    - 한번도 등장하지 않은 데이터라면 결과가 0이 됨

### 