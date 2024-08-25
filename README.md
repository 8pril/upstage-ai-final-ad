# upstage-ai-final-ad
Chemical Process Anomaly Detection | 화학 공정 이상 탐지

# 1. 대회 소개

## 1.1. 개요

24시간 내내 운영되는 화학 공정은 이상이 발생하면 막대한 금전적 피해를 입을 수 있다. 공정 상태를 예측하고 대비책을 마련하는 것이 중요한 과제인데, 이를 위해서는 공정 데이터를 이해하고 이상 징후를 파악하는 것이 필수적이다.

본 대회는 화학 공정 데이터를 이용한 이상 탐지(anomaly detection)를 수행하여, 공정 데이터에서 비정상적인 동작을 탐지하는 것을 목표로 한다. 이를 통해 공정에서 발생할 수 있는 문제를 예측하고 대비할 수 있다.

본 대회에서 사용되는 학습 데이터와 평가 데이터는 모두 CSV 파일 형태로 제공된다. 학습 데이터로는 약 25만 개의 화학 공정 데이터가 제공되며, 이에 대응하 약 72만 개의 평가 데이터가 제공된다.

이상 탐지를 위한 알고리즘 개발은 화학 공정 분야에서 매우 중요한 과제이며, 이를 통해 공정의 안정성을 높이고 예기치 않은 문제를 예방할 수 있다는 점에서 큰 의미가 있다.

## 1.2. 평가 지표

본 대회에서는 정상과 이상에 대한 F1-Score 를 계산하여 모델의 성능을 평가한다.

- 이상인 경우 : `1`
- 정상인 경우 : `0`

사용되는 정답 Label 은 위와 같으며, 실제 정답의 정상/이상과 모델의 정상/이상을 계산하여 F1 Score 를 산출한니다.

Accuracy Score 또한 리더보드에 참고용으로 제공되나, 등수 산정은 F1 Score 만을 기준으로 한다.

## 1.3. 데이터 설명

- 학습 데이터
    
    학습 데이터에는 모두 정상 가동인 상태의 화학 공정 데이터만 포함 되어 있다. 250000 개의 row와 55개의 column 으로 구성되어 있고, column 의 데이터 타입은 `float`형태입니다.
    
    각 column의 의미는 다음과 같는다.
    
    - `faultNumber` : 정상인지, 비정상인지 나타내는 Label, 정상일 경우 '0', 비정상일 경우 '1'
    - `simulationRun` : 시뮬레이션이 실행된 Run 의 번호
        - 동일한 하나의 `simulationRun` 이 정상일 경우 `faultNumber` 가 모두 '0'
        - 반대로 하나의 `simulationRun` 이 비정상일 경우 `faultNumber` 가 모두 '1'
        - 학습 데이터에는 정상 데이터만 재합니다. 따라서 `faultNumber` 가 모두 '0'
        - 테스트 데이터에는 정상/비정상 데이터가 모두 존재한다. 따라서 `faultNumber` 가 모두 '0'인 `simulationRun`도 있고, `faultNumber` 모두 '1'인 `simulationRun`도 있다.
    - `sample`: 하나의 Run 안의 sample 번호, 학습 데이터는 한 Run 당 500 sample
    - `xmeas_*` : measurement 의 약자로, 화학 공정에서 측정된 센서 값
    - `xmv_*` : manipulated variable 의 약자로, 화학 공정에서 제어되는 값
    
    !https://aistages-api-public-prod.s3.amazonaws.com/app/Files/b0d68663-641e-4466-80e9-2607b593e0e0.png
    
- 평가 데이터
    
    평가 데이터는 학습 데이터와 동일하게 52 가지 센서 값을 가진 변수가 존재하며, 학습 데이터와 다르게 정상과 이상이 모두 존재한다.
    
    평가 데이터 중 임의의 비율에 따라 public set 과 private set 으로 나누어 구성되었다.
    
    채점을 위한 데이터 제출 시.
    
    정상일 경우 : `0`
    
    이상일 경우 : `1`
    
    로 예측하여 `output.csv` 결과물 파일을 제출한다.
    

# 2. 경진대회 수행 절차

- Step 1: 접근법 수립
- Step 2: EDA
- Step 3: 모델링

# 3. 경진대회 수행 과정

## 3.1. EDA

1. 데이터 분포 

<aside>
💡 정상 데이터로만 구성된 학습 데이터와 정상 및 이상 데이터로 구성된 평가 데이터의 분포 차이를 Histogram을 통해 파악하고자 했다.

</aside>

- Train Data
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/7092b499-f5b1-4b52-9507-8664b83b1196/Untitled.png)
    
- Test Data
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/1d32c63c-087c-4ae3-9104-a8889cc53f53/Untitled.png)
    

2. 변수 간 비교

<aside>
💡 표준화를 통해 변수 간 범위를 통일 한 다음 Line Plot을 통해 특정 변수 간 상관관계를 파악하고자 했다.

</aside>

- Red Lins: xmeas_*, Blue Line: xmv_* 변수 (아래 그래프는 xmeas_31과 mvx_5 간의 관계를 나타냄)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/74998514-cd1f-4adb-aff6-ff303e73f8c1/Untitled.png)
    

3. simulation별 데이터 독립성 확인

- 두 simulationRun(1번, 2번)을 이어 전환지점(sample==500)을 표시
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/79a92edc-c093-4462-a512-8be92629cdca/Untitled.png)
    
    ⇒ 해당 그래프를 통해 데이터가 독립적이지 않고 연속적임을 알 수 있음
    
1. FFT 비교

<aside>
💡 시계열 데이터의 주파수의 특징을 파악하기 위해 1초간의 데이터를 Sampling했다는 가정하에 FFT를 수행했다.

</aside>

- Line Plot (Train | simulationRun==1 | xmeas_1) : 정상 데이터
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/20e77454-3bef-420d-9f16-72bc542b4b26/Untitled.png)
    
- Line Plot (Test | simulationRun==438 | xmeas_1) : 이상 데이터
    - Test의 simulationRun을 438번으로 설정한 이유는 수치적으로 해당 simulationRun이 이상치라고 판단했기 때문
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/7eb79a43-5230-4c34-946b-8d5d4877d455/Untitled.png)
    
    ⇒ FFT를 통해 정상 데이터와의 주파수 성분 차이를 확인할 수 있음
    
1. 상관관계 분석
- Correlation Heatmap
    - 효율적인 시각화를 위해 Correlation Matrix의 하삼각행렬만 표시 & 낮은 상관관계(-0.3~0.3) 배제
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/480ca562-9492-4fd5-9fdf-7a5ea32e978d/Untitled.png)
    
    ⇒ 해당 Matrix로 일부 변수간의 상관관계가 존재하는 것을 확인
    
1. 이웃변수 간 관계 파악
- Parallel Coordinates Plot
    - 이웃한 변수 간의 상관 관계 및 연결 강도를 시각화한 그래프
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/02655043-54bc-4504-8724-d0a738180e02/Untitled.png)
    
    ⇒ 데이터가 전체적으로 평단함을 알 수 있음   
    
1. 차원 축소 및 시각화

<aside>
💡 평가 데이터를 샘플링한 다음 2차원 공간으로 매핑하여 데이터의 패턴 및 이상치를 시각적으로 식별하고 이해하고자 했다.

</aside>

- PCA
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/1799c33d-f1dc-47e5-867c-d30b62cc1204/Untitled.png)
    
- T-SNE
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/f552e429-1834-406b-b0f6-35871aa80c9f/Untitled.png)
    

## 3.2. 모델링

### P**CA by SVDD**

- 진행 순서
    1. PCA의 결과를 통해 SVDD(구를 활용하여 정상치와 이상치를 구별하는 방식)을 고안
    2. PCA 결과를 데이터로 사용
    3. PCA의 데이터에 min, max를 활용하여 정상치와 이상치를 나누는 범위를 설정
    4. 차원 축소 수치만으로 이상치 구별
- 차원을 늘려가며 실험
    - 1차원 - Public F1 0.8269
    - 2차원 - Public F1 0.8899
    - 3차원 - Public F1 0.8908
    - 4차원 - Public F1 0.8908
    - 5차원 - Public F1 0.8870

### **OCSVM (One Class Support Vector Machine)**

- feature selection
    - 3차원 PCA 결과 각 주성분의 고유벡터를 토대로 데이터의 변동성을 가장 잘 설명하는 변수 선택
        
        ⇒ xmeas_7, xmeas_10, xmeas_13, xmeas_16, xmv_10, xmv_13
        
- 하이퍼파라미터
    - 직접 튜닝하여 성능 개선
        
        ⇒ kernel = rbf, gamma = auto. nu = 0.05, shrinking = FALSE, max_iter = 123
        

### IF (Isolation Forest)

- Holdout 8:2 비율로 검증 데이터셋 분할
    
    <aside>
    💡 Optuna Optimization을 위해 Validation Dataset(50,000)을 생성했다.
    
    Train Dataset이 모두 정상데이터(‘0’)만 존재하므로, 정상 데이터로 학습 후 Validation Dataset으로 추론 시 정상으로만 Pradiction 되어야 한다고 가정하고 진행했다.
    
    </aside>
    
    - Train
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/cf066bbf-e9e4-4521-8247-a2a30bbed3eb/Untitled.png)
        
    - Validation
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/72589cf8-3b7f-4408-bf35-c5a04386df57/Untitled.png)
        
- Optuna를 이용한 하이퍼파라미터 튜닝
    - Scoring은 valid_y(faultNumber가 전부 0)과 pred_y 간의 Accuracy로 수행
    - Accuracy가 커지도록 하이퍼 파라미터 튜닝 진행
- 후처리
    
    <aside>
    💡 하나의 simulationRun 내 sample들은 faultNumber는 모두 같아야 한다. 
    따라서 sample 단위로 prediction 되어있는 faultNumber를 simulationRun 단위로 바꿀 필요가 있다. 이러한 후처리를 위해 simulationRun을 정상/비정상으로 판단하기 위한 threshold를 설정하였다.
    
    </aside>
    
    - threshold를 1 ~ 150 혹은 simulationRun별 anomaly 개수의 평균으로 조절해가며 실험 진행

### **KNN (K-Neighbor Nearest)**

<aside>
💡 KNN은 지도학습이기에 IF(Isolation Forest)를 통한 pred 값을 y 클래스로 라벨링하여 진행했다.

</aside>

- 진행 순서
    1. PCA를 통한 차원 축소(2차원) 진행
    2. KNN(K=5)을 통한 anomaly score 계산
    3. 각 포인트별 average distance(평균거리) 및 0.1의 임계값(threshold)을 기준으로 상위 0.1%를 이상치로 판단
- 거리 척도 변경 - 3가지 방법
    - 'Euclidean', 'Manhattan', 'Minkowski'으로 실험
        
        ⇒ 거리별 이상치의 개수 차이는 발생하지 않음
        

### **K-Means**

<aside>
💡 클러스터링을 위한 비지도학습 방법인 K-Means를 사용해 정상치 클러스터의 바운더리를 계산하여 계산된 바운더리를 기준으로 이상치를 분류하기로 했다.

</aside>

- 진행 순서
    1. 정상치로만 이루어진 학습 데이터로 k-means 수행하여 클러스터의 중심 좌표(centroids) 확보
    2. 중심 좌표마다 정상치의 최대 거리를 계산하여 바운더리 계산
    3. 추론 시 중심 좌표와의 거리를 계산하여 바운더리를 벗어날 경우 이상치로 판단
- SimulateRun 단위로 최소 정상치 허용 샘플 개수 조절
    - threshold가 1일 때(이상치가 1개 이상인 SimulateRun을 모두 이상치로 판단) 시 가장 높은 성능 기록
        
        ⇒ Public F1 0.9204 / Private F1 0.9000
        
- PCA 활용
    - PCA로 차원 축소한 변수만을 사용해 K-Means 수행  ⇒ F1 스코어 하락
    - PCA로 차원 축소한 변수를 파생변수로 사용해 K-Means 수행 ⇒ F1 스코어 하락

### **AutoEncoder**

<aside>
💡 정상 데이터로 학습된 오토인코더 모델을 사용하여 입력 데이터의 정상 패턴을 학습하고 재구성 오차를 기준으로 이상치를 판별하는 방식을 시도했다. 이 방식은 데이터의 비선형적인 구조와 고차원 특징을 잘 파악할 수 있으며 특히 정상 데이터의 분포가 복잡한 경우에 유용하다.

</aside>

- 진행 순서
    1. **정상 데이터로 학습**: 먼저 정상 상태의 데이터로 오토인코더를 학습한다.
        
        *오토인코더는 입력 데이터를 압축(인코딩)하여 저차원 표현으로 매핑한 다음 다시 원래 입력으로 복원(디코딩)하는 네트워크 구조
        
    2. **재구성 오차 계산**: 학습된 오토인코더를 사용하여 입력 데이터를 재구성하고, 원본 입력과 재구성된 입력 간의 재구성 오차를 계산한다. 이 오차는 입력 데이터와 재구성 데이터 간의 차이를 나타낸다.
    3. **Anomaly Score 계산**: 재구성 오차를 사용하여 각 데이터 포인트의 Anomaly Score를 계산한다. *Anomaly Score는 재구성 오차가 큰 데이터 포인트일수록 높으며, 임계값을 초과하는 데이터 포인트는 이상치로 분류될 수 있다.
    4. **이상치 탐지**: Anomaly Score가 높은 데이터 포인트를 이상치로 분류한다.
- Latent Space를 늘려 가며 실험
    - Latent Space 2 / Layer 3 → train loss 0.65 / Public F1 0.1320
    - Latent Space 4 / Layer 3 → train loss 0.05 / Public F1 0.5228
    - Latent Space 8 / Layer 3 → train loss 0.4 / Public F1 0.4291

### 모델 앙상블

- 시도했던 모델 중 성능이 가장 좋았던 OCSVM, PCA, K-means의 추론 결과를 Hard Voting 방식으로 앙상블 진행
    
    ⇒ Public F1 0.9099 / Private F1 **0.9091**
    

# 4. 경진대회 결과

## 4.1. 리더보드 순위

- Public 리더보드
    - 1위, F1 score: 0.9204

![스크린샷 2024-05-10 오후 11.27.25.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/b596a570-4a1d-4bab-a00e-2bb323fbce99/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-05-10_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.27.25.png)

- Private(최종) 리더보드
    - 1위, F1 score: 0.9091

![스크린샷 2024-05-10 오후 11.27.51.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/e7853e40-91e4-4843-ad45-e461c2d497e8/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-05-10_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.27.51.png)

## 4.2. 결과 분석

### 스코어 향상에 도움이 된 핵심 전략

- **다양한 모델 실험 및 앙상블**
    - 강의에서 다룬 각 모델을 대회 데이터 특성에 맞추어 시도해본 뒤 가장 성능이 좋았던 모델을 앙상블하여 최종 스코어를 기록할 수 있었다.
- 학습 데이터 활용
    - 기본적으로 이상탐지를 위한 머신러닝 모델은 비지도 학습 기반의 알고리즘이기 때문에 평가 데이터를 바로 학습에 사용하여 정상 패턴과 이상치를 구별하는 방식만을 시도할 수도 있었다. 하지만 정상치에 대한 레이블 정보가 주어진 학습 데이터를 최대한 활용하기 위해서 학습 데이터로 모델을 학습시켜 정상 패턴의 경계를 파악한 다음 경계를 벗어난 데이터를 이상치로 분류하는 접근 방식 또한 시도해볼 수 있었다. 결과적으로 해당 방식으로 학습한 K-means 모델이 높은 성능을 보였다.

### 경진대회 진행 소감

- 실제 이상 탐지 문제에서는 이상 정보가 부족한 상황이 대부분이므로, 이번 대회에서도 이상치가 존재하지 않는, 즉 이상치에 대한 레이블 정보가 없는 데이터가 학습 데이터로 주어졌다. 따라서 실제와 유사한 문제에 맞는 접근 방식을 고민해볼 수 있는 좋은 기회였다.
