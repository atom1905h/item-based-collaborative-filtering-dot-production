# item-based-collaborative-filtering-dot-production

## 1. 프로젝트 소개

이 프로젝트는 Item-Based Collaborative Filtering을 사용하여 영화 추천 시스템을 구축하는 것을 목표로 합니다. 해당 시스템은 사용자가 이전에 평가한 영화와 유사한 영화를 추천함으로써 개인화된 추천을 제공합니다. 프로젝트는 다양한 전처리, 벡터화, 결측치 대치, 유사도 함수 및 추천 방식을 활용하여 구성됩니다.

## 2. 모델 평가지표

프로젝트에서 사용된 모델의 성능을 평가하기 위해 다음 평가지표를 사용합니다.

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## 3. 전처리(preprocessing.ipynb)

### 3.1 영화 제목 및 연도 처리

- 영화 제목이 같지만 movie ID가 다른 경우를 고려
- 영화 제목에 연도가 없는 경우 채워줌
- 영화 제목이 같지만 연도가 다른 경우를 고려

### 3.2 벡터화(vectorization.py)

#### 3.2.1 Movie - User 행렬

- 평점 정보
- 각 사용자의 평균 평점과 각 영화에 대한 사용자 평점과 평균과의 차이
- 영화 개봉년도와 평점 매긴 시간과의 차이
- 평점 여부
- 태그 여부
- 평점 및 태그 여부
- 사용자가 각 영화에 매긴 태그 개수
- 사용자 행동의 군집 (KMeans, DBSCAN)
- Tree 모델을 통한 Feature Combine

#### 3.2.2 Movie - Tag 행렬

- TF-IDF를 이용한 벡터화

### 3.3 결측치 대치(imputation.py)

- 0으로 대치
- 평균 (사용자 및 영화 기준)
- 중앙값 (사용자 및 영화 기준)
- 최빈값 (사용자 및 영화 기준)
- KNN을 이용한 결측치 대치

## 4. 유사도 함수(similarity.py)

- Cosine 유사도
- Euclidean 거리
- Manhattan 거리
- Jaccard 유사도
- Pearson 상관계수
- Mean Squared Difference (MSD)
- JMSD (Jaccard를 이용한 MSD)
- PSS (Proximity Significance Singularity)
- JPSS (Jaccard를 이용한 PSS)

## 5. Dot Product를 통한 평점 예측(predict.py)

- 유사도 함수를 사용하여 Item 간 유사도 계산
- Dot Product를 통해 평점 예측

## 6. 추천 방식 (Top-K)

### 6.1 기본 추천

- 사용자가 이전에 평가한 영화와 가장 유사한 영화를 추천

### 6.2 인기도 고려 추천

- 영화의 인기도 (Steam Rating, num users 등)를 고려하여 추천

## 7. 사용법
### 7.1 parser
  - --vecorization : 벡터화 함수 이름
  - --imputation : 결측치 대치 함수 이름
  - --similarity : 유사도 함수 이름
  - --weight : 가중치 적용 여부
  - --weight_sd : 가중치 적용 요소
### 7.2 example
  ```sh
python main.py --vectorization item_rating --imputation fill_median_1 --similarity jaccard --weight True --weight_sd steam_rating
```

