# TOA using generative and predictive models
Seek your fortune!: Technological impact-guided technology opportunity analysis using generative-predictive machine learning models

### 프로젝트 배경 및 필요성
- 기존 유망 기술 기회 발굴 연구들은 현재 존재하지 않는 새로운 기술 아이디어를 창출하는데 효과적이었으나, 실무에서의 활용은 제한적이었음
- 기술의 영향력은 해당 기술이 어떤 분야에 적용되는지에 따라 달라지며, 이는 기술 적용분야의 변화가 또다른 기술 기회로 이어진다는 점을 시사함
- 이에 따라, 보유기술의 기술적 영향력을 강화하고 활용성을 증대하기 위해, 기존 기술의 새로운 적용분야를 식별하는 방법론의 개발이 요구되고 있음

### 프로젝트 목표
- 머신러닝 생성 모델과 예측 모델의 통합 구조를 바탕으로, 기존 기술이 더 큰 기술적 영향력을 가질 것으로 예상되는 새로운 적용분야를 발굴하는 방법론을 개발하는 것을 목표로 함

### 데이터 – 특허 문서 데이터(USPTO)
- 특허분류코드(IPC) 집합 → 기술의 적용분야
- 특허 청구항 텍스트 → 기술 기능
- 특허 피인용 수 → 기술적 영향력 수준
  - 특정 기술영역 내 상위 10% 피인용 수를 기준으로 L1 (혁신기술)과 L2(일반기술)로 구분

### 분석 방법 – 생성-예측 통합 모델 구조
- VAE (Variational auto-encoder)와 MLP (Multilayer perceptron)을 결합하여 구성함
  - VAE는 특허분류코드 집합을 입력받는 RNN (Recurrent neural network) 인코더, 특허 청구항 텍스트를 입력받는 Transformer 인코더, 입력과 동일한 특허분류코드 집합을 생성하는 RNN 디코더로 구성되며, 이를 통해형성되는 잠재 공간을 기술 지형(Technology landscape)으로 간주함
  - MLP는 VAE를 통해 산출되는 잠재 벡터를 입력받아 입력된 기술의 기술적 영향력 수준을 예측함
- VAE와 MLP는 학습 과정에서 통합 손실 함수를 통해 함께 훈련(joint training)되어, 기술 지형에 임베딩되는 각 기술들이 기술적 영향력에 따라 유사한 영역에 위치하도록 기술 지형을 구조화함

![image](https://github.com/glee2/TOA-using-generative-predictive-models/assets/18283601/ad47485a-303b-46f5-9099-88a3732a9da0)

### 분석 방법 – 기술 지형 탐색
- 생성-예측 통합 모델의 학습이 완료된 후, 구조화된 기술 지형을 Gradient ascent search 알고리즘을 활용하여 탐색함으로써 기존 기술이 더 큰 기술적 영향력을 나타낼 것으로 예상되는 새로운 적용분야를 유망 기술 기회로 식별함

![image](https://github.com/glee2/TOA-using-generative-predictive-models/assets/18283601/d81959dc-a8ff-45b9-8734-5d47a798d36f)

### 성능 검증 실험 결과
- 모델의 신뢰성 평가
  - Jaccard similarity를 통해 모델의 특허분류코드 집합 생성 성능을 평가함
  - 이진분류에 대한 Accuracy, Precision, Recall, F1-score를 통해 모델의 기술적 영향력 수준 예측 성능을 평가함

![image](https://github.com/glee2/TOA-using-generative-predictive-models/assets/18283601/481fa0ab-8162-428c-b310-d266e7c0174e)

### 성능 검증 실험 결과
- 모델의 실용성 평가
  - 모델을 통해 식별된 적용분야의 전환이 실제로 나타나는지, 그런 경우 기술적 영향력의 변화가 있었는지를 기존 기술의 특허 인용 관계를 바탕으로 확인함

![image](https://github.com/glee2/TOA-using-generative-predictive-models/assets/18283601/4c3851cb-2250-4686-88fe-3301e9c8b2d1)
