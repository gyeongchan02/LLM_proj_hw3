# P1 Pipeline — Implementation Detail

`pipeline_code/` 안의 코드를 이해하고 실행하기 위한 가이드.
P2·P3 담당자가 코드를 수정하지 않고 작업할 수 있도록 인터페이스와 확장 지점을 명시한다.

---

## 모델 구성

| 역할 | 모델 | 비고 |
|------|------|------|
| Main Agent (에이전트) | `gpt-5.4-nano` | 범용 task 처리, 다수 호출 |
| Critic — Ours | `gpt-5.4-mini` | 판정 전용, mutating step만 호출 |
| Critic — SABER | `gpt-5.4-nano` | main agent와 **동일 모델** (ablation 기준) |
| User Simulator | `gpt-5.4-nano` | tau-bench 내장, 고객 역할 |

API 키 하나(`OPENAI_API_KEY`)만 필요하다.

---

## 1. 디렉토리 구조

```
pipeline_code/
├── src/
│   ├── critic/
│   │   ├── schemas.py          # Decision, StepLog, AgentRunResult 타입
│   │   ├── prompts.py          # 4조건 프롬프트 (Ours) + SABER 단순 프롬프트
│   │   └── critic.py           # critique_ours / critique_saber / critique_oracle
│   ├── agents/
│   │   ├── gated_env.py        # [핵심] mutating 호출 전 critic 개입
│   │   ├── vanilla.py          # Vanilla (tau-bench ToolCallingAgent 래핑)
│   │   ├── critic_agent.py     # CriticGatingAgent — Ours
│   │   ├── saber_agent.py      # SABERGatingAgent — SABER baseline
│   │   ├── oracle_agent.py     # OracleGatingAgent — Oracle 천장
│   │   ├── reflexion_agent.py  # ReflexionAgent — Reflexion baseline
│   │   └── __init__.py         # get_agent(method, **kwargs) 팩토리
│   ├── data/
│   │   └── action_taxonomy.py  # is_mutating / is_reversible / get_policy_text
│   ├── harness/
│   │   └── runner.py           # run_experiment(config) + CLI
│   └── eval/
│       └── metrics.py          # compute_all_metrics + CLI
├── configs/
│   └── experiment.yaml
├── scripts/
│   └── run_experiment.sh
├── requirements.txt
└── .env.example
```

---

## 2. 설치

```bash
# tau-bench
pip install git+https://github.com/sierra-research/tau-bench.git

# 파이프라인 의존성
cd pipeline_code
pip install -r requirements.txt

# API 키
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## 3. 실행 방법

### smoke test (vanilla 3개)
```bash
cd pipeline_code
./scripts/run_experiment.sh --method vanilla --end-index 3
```

### 단일 method
```bash
./scripts/run_experiment.sh --method ours --end-index 20
```

### 전체 5-way 비교
```bash
./scripts/run_experiment.sh --config configs/experiment.yaml
```
`end_index: null` 로 설정하면 전체 test set(~115개).

### 결과 + 지표 한 번에
```bash
./scripts/run_experiment.sh \
  --method ours saber vanilla reflexion \
  --end-index 30 \
  --compute-metrics \
  --gold-labels ../data/labels/perturbations.jsonl \
  --divergence-file ../data/labels/divergences.jsonl
```

### 지표만 계산 (JSONL 이미 있을 때)
```bash
python -m src.eval.metrics \
  --results-dir data/results \
  --gold-labels data/labels/perturbations.jsonl \
  --divergence-file data/labels/divergences.jsonl \
  --output data/results/metrics.csv
```

---

## 4. 핵심 코드 흐름

### Ours (CriticGatingAgent)

```
runner.py
  └─ CriticGatingAgent.run(task, env)
        └─ GatedEnv.step(action)
              ├─ is_mutating(tool_name)?
              │    NO  → env.step(action)            # 조회는 통과
              │    YES →
              │         critique_ours(
              │           tool, args, goal, history,
              │           policy_text,
              │           critic_model="gpt-5.4-mini"
              │         )
              │         → Decision(verdict, reason, revised_args, ...)
              │              "approve"   → env.step(action)
              │              "revise"    → env.step(revised_args) + feedback 주입
              │              "block"     → 실행 안 함, 에이전트에게 이유 전달
              │              "ask_user"  → 실행 안 함, 고객에게 질문 전달
              └─ StepLog 기록 (step, tool, args, is_mutating, decision, executed)
```

### SABER (SABERGatingAgent)

Ours와 구조 동일, 두 가지만 다름:
- critic 모델: `gpt-5.4-nano` (main agent와 **동일**)
- 프롬프트: 4조건 없이 "이 행동 맞는지 체크해" 단순 문장

### Reflexion (ReflexionAgent)

```
loop (최대 3회):
  GatedEnv(critique_fn=None)          # gating 없음
  tau-bench 에이전트 run → reward
  if reward > 0: break
  reflection = gpt-5.4-nano로 실패 반성문 생성
  task instruction 앞에 반성문 prepend
  재시도
```

### Oracle (OracleGatingAgent)

Ours와 구조 동일, critic 함수만 다름:
- LLM 호출 없음
- P2의 `perturbations.jsonl`에서 `(task_index, tool)` 키로 gold verdict 조회

---

## 5. 결과 JSONL 포맷

`data/results/{method}_seed{N}.jsonl` — 한 줄 = 한 task.

```json
{
  "task_index": 12,
  "reward": 1.0,
  "step_logs": [
    {
      "step": 1, "tool": "get_order_details",
      "args": {"order_id": "O123"},
      "is_mutating": false, "reversible": null,
      "decision": null, "executed": true, "rolled_back": false
    },
    {
      "step": 2, "tool": "cancel_pending_order",
      "args": {"order_id": "O123", "reason": "no longer needed"},
      "is_mutating": true, "reversible": true,
      "decision": "approve", "executed": true, "rolled_back": false
    }
  ],
  "metadata": {
    "method": "ours",
    "model": "gpt-5.4-nano",
    "critic_model": "gpt-5.4-mini",
    "num_blocked": 0, "num_revised": 1, "num_ask_user": 0,
    "elapsed_s": 4.2, "seed": 42
  }
}
```

---

## 6. 설정 (`configs/experiment.yaml`) 변경 지점

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `end_index` | `10` | 태스크 수 (전체 = `null`) |
| `seeds` | `[42]` | 시드 리스트 (pass^k는 여러 개) |
| `model` | `gpt-5.4-nano` | Main agent 모델 |
| `critic_model` | `gpt-5.4-mini` | Ours critic 모델 |
| `oracle_label_file` | `data/labels/perturbations.jsonl` | P2 레이블 경로 |

Ablation: yaml 하단 주석 처리된 `condition_ablation` 블록 언커멘트.  
제거 가능한 조건: `"GOAL"` / `"STATE"` / `"CONSTRAINT"` / `"POLICY"`

---

## 7. P2 확인 사항 (`action_taxonomy.py`)

tau-bench 설치 후 실제 도구명을 확인:
```python
from tau_bench.envs import get_env
env = get_env("retail", ...)
print([t["function"]["name"] for t in env.tools])
```
다른 이름이 있으면 `RETAIL_READONLY`, `RETAIL_REVERSIBLE`, `RETAIL_IRREVERSIBLE` 집합만 수정.  
**함수 시그니처 `is_mutating()`, `is_reversible()`, `get_policy_text()`는 변경 금지.**

---

## 8. Oracle 레이블 연결 (P2 → pipeline)

파일: `data/labels/perturbations.jsonl`
```json
{"task_index": 12, "tool": "cancel_pending_order",
 "gold_decision": "block", "reversible": true,
 "revised_args": null, "question_to_user": null}
```
**필수:** `task_index`(int), `tool`(str), `gold_decision`(str), `reversible`(bool)

---

## 9. Divergence 레이블 연결 (P2 → P3)

파일: `data/labels/divergences.jsonl`
```json
{"task_index": 7, "divergence_step": 3, "tool": "cancel_pending_order"}
```
`--divergence-file` 인자로 `metrics.py`에 넘기면 `recovery_rate` 자동 계산.

---

## 10. tau-bench 호환성 — 수정이 필요할 수 있는 두 곳

**A) `gated_env.py` → `_parse_action()` / `_make_action()`**  
tau-bench Action 타입 포맷이 다르면 이 두 함수만 수정.

**B) `runner.py` → `_get_tasks()`**  
버전에 따라 `env.tasks` 또는 `env.get_all_tasks()` 이름이 다를 수 있음.

---

## 11. 구현 상태

**완료 (Core)**
- [x] 4조건 구조화 프롬프트 + SABER 단순 프롬프트
- [x] `critique_ours()` — gpt-5.4-mini, JSON 파싱 + fallback
- [x] `critique_saber()` — gpt-5.4-nano (동일 모델)
- [x] `critique_oracle()` — P2 레이블 파일 조회
- [x] `GatedEnv` — 4-way 판정, revise 재시도, block/ask_user 차단
- [x] 5개 메서드 (vanilla / ours / saber / oracle / reflexion)
- [x] `runner.py` + `metrics.py` + `run_experiment.sh`

**미구현 (Stretch)**
- [ ] rollback-to-divergence 실제 구현 (훅만 있음)
- [ ] fine-tuned critic (SFT)
