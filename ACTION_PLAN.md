# ACTION PLAN

환경 셋업부터 영상 제출까지의 실행 계획. 각 단계는 산출물과 완료 기준(Done)을 갖는다.

- **최종 산출물:** 발표 영상(15분 이내).
- **마감:** 2026-06-21.

---

## 0. Scope

| 티어 | 포함 |
|------|------|
| **Core (반드시)** | 분리 critic(다른 계열) · 4대 조건 프롬프트 · 4-Way Gating · 재프롬프트 회복 · ask_user · 정답 액션 변형 라벨 · 분기점 라벨 · 메인 비교 + 기본 ablation · oracle critic |
| **Stretch (시간 남으면)** | rollback-to-divergence · cross-family ablation 전체 · fine-tuned critic · hidden-state 스크리닝(RepE) · airline · tau²-bench |

- **벤치마크:** 주력 = tau-bench retail. airline·tau² = stretch.
- **데이터 생성:** SFT 여부와 무관하게 Core(회복률·critic 정확도 측정에 필요).
- **SFT:** stretch. 먼저 프롬프트 critic으로 완성하고 시간 되면 학습 critic 추가.
- **파이프라인:** 데이터와 병렬로, 프롬프트 critic 임시 버전으로 먼저 만든다.

---

## 역할 분담 (4인)

| 담당 | 영역 | 시작 | 의존성 |
|------|------|------|--------|
| **P1 (파이프라인)** | 파이프라인 + Critic + baseline(vanilla/Reflexion/SABER) + oracle | 즉시 | 없음(임계 경로) |
| **P2 (데이터)** | 분류표 · 정답 액션 변형 생성기 · 분기점 라벨러 | 즉시 | 없음 |
| **P3 (실험)** | metric 코드 · 반복 실행 runner · 정량 분석 | 즉시(metric·runner 먼저) | 실행은 P1·P2 완성 후 |
| **P4 (영상)** | 영상 + 결과 시각화·error analysis 사례 정리 | 빌드 후반 | 결과 산출 후 |


핵심: 파이프라인은 데이터를 기다리지 않는다. 데이터가 필요한 시점은 "평가"부터.

---

## Phase 0 — 셋업 (전원 공유)

```bash
# tau-bench 설치
pip install git+https://github.com/sierra-research/tau-bench.git

# 파이프라인 의존성
cd pipeline_code && pip install -r requirements.txt

# API 키 (.env 파일)
echo "OPENAI_API_KEY=sk-..." > .env
```

smoke test (vanilla 3개):
```bash
./scripts/run_experiment.sh --method vanilla --end-index 3
```
**Done:** `data/results/vanilla_seed42.jsonl` 생성, 에러 없음.

---

## 팀 인터페이스 계약 (P1 ↔ P2 ↔ P3)

> **핵심 원칙:** `pipeline_code/` 안의 코드는 수정하지 않아도 된다.
> P2는 레이블 파일만 만들고, P3는 YAML 설정과 CLI만 쓴다.

### P2가 만들어야 하는 파일 2종

#### ① 정답 액션 변형 레이블 (Oracle + Critic 정확도 측정용)
파일 경로: `data/labels/perturbations.jsonl`  
포맷 (한 줄 = 한 판정 포인트):
```json
{"task_index": 12, "tool": "cancel_pending_order",
 "args": {"order_id": "O123", "reason": "no longer needed"},
 "gold_decision": "block",
 "reversible": true,
 "revised_args": null,
 "question_to_user": null}
```
**필수 필드:** `task_index`(int), `tool`(str), `gold_decision`(str), `reversible`(bool)  
**선택 필드:** `args`, `revised_args`, `question_to_user`  
**`gold_decision` 값:** `"approve"` | `"block"` | `"revise"` | `"ask_user"`  
**주의:** `task_id` 문자열이 아닌 **정수 `task_index`** 로 키를 설정해야 함
(tau-bench task 순서 index와 동일)

#### ② 분기점 라벨 (Recovery Rate 측정용)
파일 경로: `data/labels/divergences.jsonl`  
포맷:
```json
{"task_index": 7, "divergence_step": 3, "tool": "cancel_pending_order"}
```
**필수 필드:** `task_index`(int), `divergence_step`(int)  
P3가 `--divergence-file data/labels/divergences.jsonl` 인자로 metrics.py에 넘기면 자동 계산됨.

#### ③ `action_taxonomy.py` 도구명 검증 (P2 책임)
`pipeline_code/src/data/action_taxonomy.py`에 사전 분류된 retail 도구 목록이 있다.
**아래 명령으로 실제 tau-bench 도구명을 확인하고 다르면 P2가 해당 집합을 수정한다:**
```python
from tau_bench.envs import get_env
env = get_env("retail", ...)
print([t["function"]["name"] for t in env.tools])
```
수정 가능한 것: `RETAIL_READONLY`, `RETAIL_REVERSIBLE`, `RETAIL_IRREVERSIBLE`  
수정하면 안 되는 것: 함수 시그니처 `is_mutating()`, `is_reversible()`, `get_policy_text()`

---

### P3가 실험을 돌리는 방법

#### 전제 조건
1. Phase 0 셋업 완료
2. `pipeline_code/.env`에 `OPENAI_API_KEY` 설정

#### 메인 비교 실행 (5-way)
```bash
cd pipeline_code
./scripts/run_experiment.sh --config configs/experiment.yaml
```

결과는 `data/results/{method}_seed42.jsonl` 에 저장됨.

#### 지표 계산
```bash
python -m src.eval.metrics \
  --results-dir data/results \
  --gold-labels ../data/labels/perturbations.jsonl \
  --divergence-file ../data/labels/divergences.jsonl \
  --output data/results/metrics.csv
```

#### Ablation 실험
`configs/experiment.yaml`의 주석 처리된 `condition_ablation` 항목 언커멘트.  
제거 가능한 조건: `"GOAL"` / `"STATE"` / `"CONSTRAINT"` / `"POLICY"`

#### pass^k 실험
`configs/experiment.yaml`에서 `seeds: [42, 7, 123]`으로 설정 후 동일 명령 실행.  
`metrics.py`는 동일 task_index의 여러 시드 결과를 합산해 pass^k를 계산한다.

---

## Phase 1 (P2) — 데이터/라벨 생성

### 1.1 도구·가역성 분류표 — `src/data/action_taxonomy.py`
도구를 mutating/조회, 가역/비가역으로 분류(실제 도구명은 repo 확인).

| 분류 | 예시(retail) | Critic | 가역성 |
|------|------|--------|--------|
| mutating | `cancel_pending_order`, `modify_pending_order_items` | O | 확정 전 = 가역 |
| mutating | 결제·발권 확정류 | O | 비가역 |
| read-only | `get_*`, `find_*`, `list_*` | X | 해당 없음 |

### 1.2 정답 액션 변형 — `src/data/perturb.py` (주력)
정답 액션을 변형해 라벨된 오답 생성. 인자 오류→`revise`, 정책 위반→`block`, 정보 부족→`ask_user`, 변형 없음→`approve`. JSONL 예:
```json
{"task_id":"retail_012","step":7,"tool":"modify_pending_order_items","args":{"...":"..."},"gold_decision":"revise","reversible":true}
```

### 1.3 분기점 라벨러 — `src/eval/label_divergence.py` (보완)
같은 task의 성공·실패 로그를 정렬해 처음 갈라지는 mutating step을 표시. 회복률 측정의 기준. P3가 돌린 실제 로그에 적용한다.

> 누수 금지: 이 라벨은 평가·학습에만, 실행 중 프롬프트엔 넣지 않는다.

**Done:** 분류표 확정, 변형 생성기·분기점 라벨러 동작, 라벨 로더.

데이터 포맷·활용 방법은 이 파일 상단의 **팀 인터페이스 계약** 섹션을 참조.

---

## Phase 2 (P1) — Critic 모듈 ✅ 구현 완료

### 2.1 스키마 — `src/critic/schemas.py`
```python
Verdict = Literal["approve","block","revise","ask_user"]

@dataclass
class Decision:
    verdict: Verdict
    reason: str
    revised_args: Optional[dict] = None
    question_to_user: Optional[str] = None
    reversible: Optional[bool] = None
    rollback_to_step: Optional[int] = None
```

### 2.2 구조화 프롬프트 — `src/critic/prompts.py`
4대 조건을 명시하고 JSON으로만 답하게 강제. 판정 규칙(approve/revise/block/ask_user)을 명문화하고, 불확실하면 ask_user로 둔다. 도메인 정책 전문을 넣는 것이 정책 대조의 핵심이며 ablation에서 조건을 하나씩 뺀다.

### 2.3 호출·파싱 — `src/critic/critic.py`
- 본체와 다른 계열을 `CRITIC_MODEL`로 사용.
- 파싱 실패/타임아웃 fallback = `ask_user`.
- (Stretch) `repe.py`: hidden-state 1차 위험도 점수 → 임계 초과 시에만 full critic 호출.

**Done:** `pipeline_code/src/critic/` — schemas.py, prompts.py, critic.py 구현 완료.
4가지 verdict + fallback 포함. 단위 테스트는 각 critique_* 함수를 직접 호출해 확인.

---

## Phase 3 (P1) — 통합: Gating + Recovery ✅ 구현 완료

`src/agents/critic_agent.py` — 기존 루프를 감싸 mutating 직전에 critic을 끼운다.
```python
proposed = agent.next_action(obs)
if not is_mutating(domain, proposed.tool):
    result = env.step(proposed)                      # 조회는 통과
else:
    if STRETCH_ROLLBACK: ckpt = snapshot(env)
    d = critique(goal, state, policy, proposed)
    if   d.verdict == "approve":  result = env.step(proposed)
    elif d.verdict == "revise":   result = retry_with(d.revised_args, reason=d.reason)
    elif d.verdict == "ask_user": result = ask_user(d.question_to_user)
    elif d.verdict == "block":
        if d.reversible and STRETCH_ROLLBACK:
            restore(env, d.rollback_to_step or ckpt)
        result = feedback("BLOCKED: " + d.reason)
```
- 재발 방지: block/revise 시 사유·이전 실수 요약을 다음 관측에 주입.
- 무한 루프 방지: 동일 action 재시도 상한(예: 2회) 초과 시 ask_user 또는 종료.
- 로깅: 매 step `{tool, args, is_mutating, reversible, decision, executed?, rolled_back?}`.

**Done:** `pipeline_code/src/agents/gated_env.py` + `critic_agent.py` 구현 완료.
검증: `./scripts/run_experiment.sh --method ours --end-index 1` 실행 후 JSONL의 step_logs에 decision 필드 확인.

---

## Phase 4 (P1) — Baseline + Oracle ✅ 구현 완료

동일 하니스 위에서 비교(파이프라인과 병렬로 미리).
- **Vanilla:** 보호장치 없는 tool-calling agent.
- **Reflexion:** 실패 후 같은 모델이 반성문 주입(사후·동일모델).
- **SABER:** mutating 직전 같은 모델로 단순 prompting(중간·동일모델·비구조화).
- **Oracle critic:** 정답 라벨을 보고 판정하는 상한 — "완벽한 탐지기면 성능이 어디까지 오르나"를 보여줌. 우리 critic은 절대 정답을 보지 않는다.

> 풀 재현이 어려우면 핵심 메커니즘만 재현하고 영상에 명시. "사후 vs 사전", "동일모델 vs 분리" 축은 반드시 보존.

**Done:** vanilla.py, saber_agent.py, reflexion_agent.py, oracle_agent.py 구현 완료.
모두 `get_agent(method, **kwargs)` 팩토리로 동일하게 호출됨.

---

## Phase 5 (P3) — 평가 하니스 & 지표

`src/harness/runner.py`: `configs/experiment.yaml`(method × domain × seeds × split) 순회 실행, 결과 저장.

`src/eval/metrics.py`:

| 그룹 | 지표 | 정의 |
|------|------|------|
| Task | pass@1 | 1회 성공률 |
| Task | pass^k | k회 모두 성공(일관성) |
| Task | **Recovery Rate** | 분기 발생 task 중 최종 성공 — 가설 직접 검증 |
| Critic | Precision/Recall | block 판정 정확도 |
| Critic | False Block Rate | approve여야 할 것을 block한 비율 |
| Critic | 4-Way Accuracy | 라벨 대비 일치율 |
| Critic | Reversibility Accuracy | 가역성 판단 정확도 |
| 실용성 | Latency / Token Cost | critic 개입 증가분 |

> 개발 중엔 `--end-index 5~10`으로 검증, 정식 측정은 마지막에 전체로. 시드 고정·캐싱.

**Done:** `runner.py` + `metrics.py` + `run_experiment.sh` 구현 완료.
상단 인터페이스 계약의 P3 실행 명령 참조. 결과 CSV는 `data/results/metrics.csv`에 저장됨.

---

## Phase 6 — 실험 & Ablation (P3, 분석은 P4와 분담)

- **메인 비교:** retail(+airline) × {vanilla, reflexion, saber, ours, oracle}.
- **구조화 효과:** 단순 critic vs 4대 조건. + 조건 하나씩 제거해 기여도 분해.
- **Gating:** 4-way vs 2-way.
- **분리 효과(핵심):** 동일 계열 vs 다른 계열 critic. 같은 계열도 비슷하면 "두 번 본 효과"라는 반론이 성립 — 이 ablation 필수.
- **(Stretch) rollback 유무.**

**Done:** 메인 비교표 + ablation 표, 가설 지지/반증.

---

## Phase 7 — 분석 & Error Analysis (P4)

- 정량: 성공률·회복률, FBR vs Recall trade-off, latency/cost 대비 이득.
- 정성: critic이 잘 잡은/오차단한/놓친 케이스, 가역성 오판 사례를 trajectory와 함께.
- 한계 + future: 미완 stretch(fine-tune, RepE, 분기점 복원, tau²)를 다음 단계로.

---

## Phase 8 — 영상 (P4, ≤15분)

| 섹션 | 분량 |
|------|------|
| 문제·동기 | ~1.5분 |
| 관련 연구 위치짓기 | ~1.5분 |
| 방법(4대 조건·4-way·분리·회복) | ~3분 |
| 평가 설계 | ~2분 |
| 결과+ablation+error analysis+future | ~5분(가장 길게) |
| 마무리 | ~1분 |

15분 초과 금지. 미완 부분은 "한계+future"로 전환.

