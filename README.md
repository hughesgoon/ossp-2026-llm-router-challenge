# OSSP 2026: SK텔레콤 지정과제

`BPE + Retrieval + 2-Stage Allocation` 방식으로 구현한 초경량 라우터입니다.
고전 임베딩 특성상 문맥 이해력이 약하지만, 빠른 속도와 예산 안전성을 기준으로 튜닝했습니다.

## 로컬 실행 및 재현

본 레포지토리 최상단에서 아래 순서로 명령을 실행해야합니다.

> 참고: learn_bpe는 4분 이상 소요됩니다.

```bash
export OSSP_REPO=$(pwd)
export PYTHONPATH=src/placer

python -m train.learn_bpe
python -m train.build_dataset
python -m train.fit_heads
python -m train.calibrate_cost
python -m train.export_artifact

python -m eval.simulate
python -m eval.bootstrap
python -m eval.audit_determinism
```

## 배포

### 컨테이너 빌드

`container/Dockerfile.placer` 는 원본 `Dockerfile`을 그대로 따르되,
`src/placer/router` 만 이미지에 포함하고 학습·검증 코드는 제외합니다.

```bash
docker buildx build \
  --platform linux/arm64 \
  --provenance=false --sbom=false \
  -f container/Dockerfile.placer \
  -t ossp-router:check \
  --load .
```

`--provenance=false --sbom=false` 는 buildx가 기본으로 첨부하는 attestation을 꺼 단일 매니페스트 이미지로 만듭니다. 이게 없으면 일부 환경(containerd 이미지 스토어)에서 `docker run` 시 `No such image` 오류가 발생할 수 있습니다.

로컬 검증(공식 `check_runtime.py` 활용)은 과제 지시에 따라 [`docs/RUNTIME.md`](docs/RUNTIME.md)를 따릅니다. 레지스트리 배포는 아래 "이미지 배포" 절 참고.

## 성능 (Dev 기준)

| tier | 예산비율 | 판정 | score | 선택분포 (light/ax31/k1) |
| --- | ---: | --- | ---: | --- |
| fast | 1.100 / 1.25 (88%) | OK | 0.6601 | 636 / 232 / 0 |
| balanced | 1.790 / 2.0 (90%) | OK | 0.7062 | 176 / 692 / 0 |
| premium | 2.627 / 4.0 (66%) | OK | 0.7310 | 177 / 648 / 43 |

**가중 총점 0.6952** (baseline hash-regex 0.6954, oracle 0.8070)

> baseline보다 약간 낮긴 하지만, 스트레스 시나리오 테스트에서 예산을 초과하는 경우가 없는 설정값입니다.

## 라우팅 방식

- BPE와 ngram으로 feature 구성.
- 어떤 모델이 적합한지 예측 (score head)
  - 학습 프롬프트로 구축한 검색 DB(kNN)와 ngram 회귀를 결합해, 세 모델
    각각이 해당 프롬프트를 맞힐 확률을 추정.
- 예상 비용 추정 (cost head)
  - 로그 비용을 직접 회귀한 뒤 분위수 매핑으로 보정해, 실제 비용 분포에
    맞춘 값을 사용.
- k1을 써도 되는지 판별 (gate head)
  - k1 투입이 실제로 유효한 case인지 별도로 판별해, 폭주 위험이 큰
    case를 사전에 배제.
- 최종 라우팅은 k1에 할당할 소수를 먼저 배치하고, 남은 예산에서 ax31/light를 분배하는 방식.

```text
프롬프트
  -> BPE(4000 merge) 1-2gram 해싱 4096
     + 문자 2-5gram 해싱 2048 (앞 500자)
     + dense 8
     = 6152 차원
  -> score 3 head: retrieval(0.8) + ngram(0.2)
       retrieval = 지도가중 metric -> SVD 512 사영 -> kNN(K=5/20/80, T=0.05/0.2)
       ngram head 학습 타깃에만 이웃 평활 (K=10, alpha=0.3)
  -> cost 3 head: log(비용) 직접 회귀 -> quantile mapping
  -> gate head: k1 유효성 판별
  -> 2단계 배분: k1 우선(개수 상한) -> 남은 예산 ax31 fill
```

### 등급별 정책

```text
fast      k1 금지,     fill 0.85, ax31 예측이득 상위 70%
balanced  k1 금지,     fill 0.80, ax31 예측이득 상위 80%
premium   k1 5% 상한,  fill 0.55, ax31 예측이득 상위 80%
```

### 설계 흐름

기본 전제: light와 ax31의 경우 모든 문제에 대해서 유사한 모델이지만 성능적인 차이가 있고, k1은 추론으로 인해 다른 특성을 지닌다는 점에서 출발합니다.  

- `light vs ax31`은 학습셋 기준 75%가 둘 다 처리 가능한 프롬프트였지만, 특정 모델이 선호되어야 할 25%의 경우 프롬프트에서 유의미한 단서를 찾을 수 없었습니다.
- `k1`이 필요한 케이스는 프롬프트를 기반으로 어느 정도 추정이 가능했습니다.
- 다만 `k1`이 폭주하는 경우가 종종 발생했는데, 소수 판별, 소인수분해, 자기확인, 반복문 내 변수 재할당 등 추론 모델이 자기 확인을 반복하는 경우입니다.
  - 이 경우에도 모델의 특성으로 인한 폭주일 뿐, 프롬프트 상 구분은 어려웠습니다.
  - 35자 프롬프트가 31,000토큰을 쓰는 케이스가 존재합니다. (ex. `List the prime factors of 38948129.`)
  - 키워드 기반으로 대략적인 차단은 가능하나, 놓친 소수의 프롬프트만으로도 예산을 초과할 정도로 위험성이 높았습니다.
  - 점수 하락을 감수해서라도 `k1` capacity를 제약하는 방식을 채택했습니다.

| 문제 | 판별력 측정 | 대응 |
| --- | --- | --- |
| light vs ax31 | AUC 0.54~0.60 | 예측 불가. 예산 여유분으로만 사용 |
| vs k1 | gate AUC 0.775 | 예측 가능. 예산 우선 배정 |
| k1 폭주 여부 | AUC 0.762 | 부족. 개수 상한 방어 도입 |

#### 추가 1. k1 폭주 케이스와 지금 라우터의 배정 패턴

dev 기준 배정 실측 (premium, k1 총 43건):

| 패턴 | train 폭주율 | dev 폭주율 | k1 배정 |
| --- | ---: | ---: | ---: |
| (기저) | 10.0% | 10.8% | 43 |
| prime factor | 60.0% | 80.0% | **0** |
| 한국나이/띠/존댓말 | 79.2% | 77.8% | **0** |
| List the | 66.7% | 66.7% | **0** |
| count letters | 20.0% | 100.0% | **0** |
| is prime | 50.0% | 0.0% | **0** |
| 큰수(8자리+) | 19.6% | 15.4% | 12 |

명시적 규칙 없이도 gate head가 폭주 위험 패턴을 전부 배제합니다.

#### 추가 2. 폐기한 시도

| 시도 | 폐기한 이유 결과 |
| --- | --- |
| 프롬프트의 특성별 축 분류 | n-gram에 이미 포함 |
| 축별 난이도 캐스케이드 | R2 0.428 -> 0.425 |
| 문항반응이론 MIRT rank-1/2 제약 | rank-2 = 자유 head 동일, rank-1은 k1 붕괴 |
| 위치 정보 / word n-gram / IDF 활용 | 각 +0.004 이하 |
| 코드 문항 AST 정적 분석 (루프/중첩/인자길이) | 폭주 판별 AUC 변화없음 |
| 코드 문항 k1 배제 | 역효과 (예산 초과 2->7회). 코드가 k1 효율 최고 24.3점/배 |
| 코드 문항 ax31 배제 | 역효과 (-0.004). ax31 효율도 최고 308.9점/배 |
| MLP Head 학습 | ridge 대비 전 지표 열세 |
| CHAR_CUT 500 -> 4000 | score 하락 + 시간 2배 |
| static embedding | 단독 0.165, 결합 +0.005 |
| SW ranking (Bradley-Terry) | 단독 0.221, 최적 가중치 0.0 (kNN과 상관 0.95) |
| 유사한 프롬프트의 난이도 DB 라벨 통일 | 유사쌍의 96%가 지문공유+질문상이. 묶어선 안됐음. |
| 문항 내 질문이 그대로 존재하는지 정보를 활용 | 실제로 유의미한 지표이나, 라우팅 성능에 영향이 미미함. |

#### 추가 3. 문헌 사례와 설계상의 대응

RouteLLM(Ong et al., ICLR 2025), MMR-Bench, RouteJudge 중심으로 기존의 문제 확인.

- **저데이터에서 고용량 모델 실패**: RouteLLM에서 BERT/causal LLM 분류기가 Arena 데이터만으로는 무작위 수준. MLP 실패와 동일 현상
- **MF 우세 보고**: RouteLLM/MMR-Bench 모두 matrix factorization 권장.
- **IRT 라우터 존재**: RouteJudge에 NIRT/MIRT-Router 포함. 그러나 본 챌린지의 학습셋은 IRT 1차원이 깨져 2차원 이상이 필요했음.
- **instance-based 불안정 경고**: MMR-Bench 지적과 반대로, 우리 데이터에서는 retrieval 비중이 높을수록 스트레스 기대값이 좋았음 (0.0 -> 0.6759, 0.8 -> 0.6822). 축별 이질성이 커서 국소 구조가 유효
