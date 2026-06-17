# P3 실험 운영 가이드

P3(실험·평가) 담당자를 위한 문서다. 환경 설정부터 최종 지표 산출까지의 전 과정을 다룬다.  
파이프라인 구조는 `implementation_detail_p1.md`, 데이터·라벨 상세는 `implementation_detail_p2.md` 참조.

---

## 0. 사전 준비

### 0.1 환경

```bash
conda activate llm_hw3   # python 3.11 + tau-bench + 의존성
# (없으면) conda create -n llm_hw3 python=3.11
#          pip install -r pipeline_code/requirements.txt
#          pip install git+https://github.com/sierra-research/tau-bench.git
```

### 0.2 API 키

```bash
# pipeline_code/.env 생성
cp pipeline_code/.env.example pipeline_code/.env
# 파일 열어서 실제 키 입력
OPENAI_API_KEY=sk-...
```

### 0.3 모델 구성

- Main agent: `gpt-5.4-nano` (vanilla, reflexion, saber, oracle 포함 모든 방법)
- Critic (ours 전용): `gpt-5.4-mini`

`configs/experiment.yaml`에 이미 설정되어 있다. 별도 수정 불필요.

### 0.4 데이터 현황

| 파일 | 상태 | 용도 |
|------|------|------|
| `data/labels/perturbations.jsonl` | 완비 (146개) | Critic 정확도 측정 |
| `data/labels/divergences.jsonl` | **임시 (bootstrap)** | Recovery Rate 측정 — §3.2 참조 |
| tau-bench retail tasks | tau-bench 설치 시 자동 포함 | 실험 실행 |

---

## 1. 실험 전체 설계

총 **9회** 실행 (메인 5 + Ablation 4).

| 실험 | 방법 | 목적 |
|------|------|------|
| E1 | vanilla | 기준선 (Critic 없음) |
| E2 | reflexion | 사후 반성 기준선 |
| E3 | saber | 단순 Critic 기준선 (동일 모델, 비구조화 프롬프트) |
| E4 | ours | 제안 방법 (별도 모델, 4조건 구조화 Critic) |
| E5 | oracle | 이론적 상한 (gold 라벨 직접 참조) |
| A1 | ours –GOAL | Ablation: GOAL 조건 제거 시 영향 |
| A2 | ours –STATE | Ablation: STATE 조건 제거 시 영향 |
| A3 | ours –CONSTRAINT | Ablation: CONSTRAINT 조건 제거 시 영향 |
| A4 | ours –POLICY | Ablation: POLICY 조건 제거 시 영향 |

---

## 2. 실행 순서

순서가 중요하다. **E1(vanilla)을 먼저 완료한 후 divergences.jsonl을 재생성**하고, 이후 나머지를 진행한다.

```
E1(vanilla) → divergences.jsonl 재생성 → E2~E5 + A1~A4 (순서 무관) → 지표 산출
```

---

## 3. 메인 실험 (E1–E5)

### 3.1 E1–E5 일괄 실행

```bash
cd pipeline_code

# 전체 실행 (config에 정의된 5개 방법 모두)
./scripts/run_experiment.sh --config configs/experiment.yaml

# 또는 개별 실행
./scripts/run_experiment.sh --method vanilla
./scripts/run_experiment.sh --method reflexion
./scripts/run_experiment.sh --method saber
./scripts/run_experiment.sh --method ours
./scripts/run_experiment.sh --method oracle
```

**출력:** `data/results/{method}_seed42.jsonl` — task별 reward + step_logs 포함.

**smoke test (선택):** 전체 실행 전 소규모 검증.

```bash
./scripts/run_experiment.sh --method vanilla --end-index 5
```

### 3.2 divergences.jsonl 재생성 (E1 완료 직후 필수)

현재 `data/labels/divergences.jsonl`은 bootstrap 근사치로, 그대로 사용하면 Recovery Rate가 자기참조 문제를 가진다.  
**E1(vanilla) 로그를 기반으로 반드시 재생성한다.**

```bash
cd pipeline_code
python -m src.eval.label_divergence vs-gold \
    --run ../data/results/vanilla_seed42.jsonl \
    --env retail \
    --split test \
    --out ../data/labels/divergences.jsonl
```

---

## 4. Critic 정확도 측정 (E3, E4 대상)

라이브 실험과 **별개로** 오프라인 하니스를 사용한다.  
`metrics.py::compute_critic_metrics`는 사용하지 않는다 — args 매칭 이슈로 유효한 수치를 보장하지 않는다.

```bash
cd pipeline_code

# Ours critic (gpt-5.4-mini)
python -m src.eval.critic_accuracy \
    --perturbations ../data/labels/perturbations.jsonl \
    --critic ours \
    --model gpt-5.4-mini \
    --cache ../data/results/critic_cache_ours.json \
    --out ../data/results/critic_acc_ours.json

# SABER critic (gpt-5.4-nano, main agent와 동일)
python -m src.eval.critic_accuracy \
    --perturbations ../data/labels/perturbations.jsonl \
    --critic saber \
    --model gpt-5.4-nano \
    --out ../data/results/critic_acc_saber.json
```

**`--cache` 옵션:** 중단 후 재실행 시 이미 호출한 결과를 재사용하여 API 비용 절감.

---

## 5. Ablation 실험 (A1–A4)

`configs/experiment.yaml` 하단 주석을 해제하여 사용한다.

```yaml
# experiment.yaml — methods 하단에 추가
- name: ours
  critic_model: gpt-5.4-mini
  condition_ablation: ["GOAL"]

- name: ours
  critic_model: gpt-5.4-mini
  condition_ablation: ["STATE"]

- name: ours
  critic_model: gpt-5.4-mini
  condition_ablation: ["CONSTRAINT"]

- name: ours
  critic_model: gpt-5.4-mini
  condition_ablation: ["POLICY"]
```

```bash
./scripts/run_experiment.sh --config configs/experiment.yaml
```

Ablation은 `critic_accuracy.py`의 `--ablate` 플래그로도 실행 가능하다.

```bash
python -m src.eval.critic_accuracy --critic ours --ablate POLICY \
    --perturbations ../data/labels/perturbations.jsonl \
    --model gpt-5.4-mini \
    --out ../data/results/critic_acc_ours_noPolicy.json
```

---

## 6. 지표 산출 및 결과 정리

### 6.1 전체 지표 자동 산출 (CSV 자동 생성)

```bash
cd pipeline_code
python -m src.eval.metrics \
    --results-dir ../data/results \
    --gold-labels ../data/labels/perturbations.jsonl \
    --divergence-file ../data/labels/divergences.jsonl \
    --output ../data/results/metrics.csv
```

`metrics.csv`가 자동 생성된다. 터미널에도 동시 출력된다.

### 6.2 지표 정의

**Task 지표**

| 지표 | 정의 | 해석 |
|------|------|------|
| pass@1 | 전체 task 중 1회 시도 성공 비율 | 기본 성능 |
| pass^k | 동일 task k회 시도 모두 성공 비율 | 일관성 (seeds 복수 필요) |
| Recovery Rate | 실수가 발생한 task 중 최종 성공 비율 | **핵심 지표** — Critic의 회복 능력 |

**Critic 지표** (`critic_accuracy.py` 출력)

| 지표 | 정의 | 해석 |
|------|------|------|
| 4-way accuracy | approve/revise/block/ask_user 4분류 정확도 | 전반적 판정 정확도 |
| Precision (block) | block 판정 중 실제 block이어야 하는 비율 | 오탐 최소화 |
| Recall (block) | 실제 block이어야 하는 것 중 block 판정 비율 | 누락 최소화 |
| False Block Rate | 올바른 행동을 block 판정한 비율 | **낮을수록 좋음** — 정상 행동을 막으면 task 실패 |
| Reversibility accuracy | 가역/비가역 판단 정확도 | ask_user 경계 정밀도 |
| revise_arg_accuracy | revise 판정 시 제안한 인자가 gold와 일치 비율 | revise 품질 |

**실용성 지표** (metrics.py 자동 산출)

| 지표 | 정의 |
|------|------|
| mean_s / median_s | task당 평균·중앙값 소요 시간 |
| total_mutating_steps | mutating tool call 총 횟수 |
| num_blocked / num_revised / num_ask_user | Critic 판정 분포 |

### 6.3 비교 구조 (논문 테이블 기준)

| 비교 | 방법 | 검증하는 주장 |
|------|------|--------------|
| 기준선 | vanilla vs ours | Critic 개입 자체의 효과 |
| 사후 vs 사전 | reflexion vs ours | 개입 시점의 중요성 |
| 동일 모델 vs 별도 모델 | saber vs ours | 모델 분리의 효과 |
| 실제 vs 상한 | ours vs oracle | 우리 Critic이 이론적 상한에 얼마나 근접하는가 |
| 조건별 기여 | A1–A4 vs ours | 4조건 각각의 기여도 |

---

## 7. 주의사항

- `divergences.jsonl` 재생성 전에 Recovery Rate를 계산하면 수치가 무의미하다 (§3.2 참조).
- `metrics.py::compute_critic_metrics`는 사용하지 않는다. Critic 정확도는 `critic_accuracy.py`로만 측정한다.
- oracle 방법은 `data/labels/perturbations.jsonl`에 없는 행동을 approve로 처리한다. oracle의 Recovery Rate가 ours보다 현저히 높지 않으면 라벨 커버리지 문제일 수 있다.
- pass^k 측정이 필요하면 `configs/experiment.yaml`의 `seeds` 항목에 복수 seed를 추가한다 (예: `[42, 7, 123]`).
- 합의 필요 사항 잔여분: `integration_issues_p2.md` C3 (divergences 재생성)은 P3가 §3.2에서 처리한다.
