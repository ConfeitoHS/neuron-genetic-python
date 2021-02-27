# neuron-genetic-python
neuron network with genetic optimisation
## 개요
[SNN](https://doi.org/10.1016/S0893-6080(97)00011-7)의 구조를 적용해, 모든 노드가 상호 가중치로 연결된 네트워크를 Python으로 구현했습니다. Leaky Integrate and Fire 모델을 사용했습니다.<br/>
가중치 행렬을 유전 알고리즘을 통해 학습합니다.<br/>
네트워크에 특정 자극을 가한 뒤 일정 시간 이내로 운동뉴런이 반응할 수 있도록 최적화합니다.
### main.py
유전 알고리즘을 실행하는 데 필요한 함수와 학습 실행 코드가 있습니다. 실행 시 해당하는 파일을 이어서 학습하거나 새로 학습합니다.
### playground.py
학습시킨 가중치를 사용합니다. pygame으로 이루어져 있으며, 마우스를 클릭해 신경망에 자극을 주고, 반응을 관찰할 수 있습니다.
### visualizer.py
networkx를 이용해 각 뉴런 노드가 어느 정도의 강도로 연결되어 있는지 양방향 그래프로 나타냅니다. 가중치의 heatmap도 관찰할 수 있습니다.
## 기타
<code>ws_gen=118_i=16_s=4_m=4_98.npy</code>는 한 세대 당 200개의 개체가 존재하며, 118세대 진화 결과 한 세대의 평균 정확도가 98% 이상인 가중치 행렬 데이터입니다.</br>
시뮬레이션을 위해 <code>playground.py</code>에 사용된 <code>ws_ext_gen=118_acc=98.npy</code>는 위의 200개의 개체 중, 모든 자극에 대해서 올바르게 반응하는 가중치 행렬을 한 가지 선택한 것입니다.
