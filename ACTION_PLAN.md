# ACTION PLAN — Pre-Action Error Detection, Gating & Recovery

처음부터 끝까지(환경 셋업 → 영상 제출) 프로젝트를 굴리기 위한 **end-to-end 실행 계획**입니다. 각 Phase는 산출물(deliverable)과 완료 기준(Done)을 갖습니다. Claude Code에 작업을 넘길 때 이 순서대로 시키면 의존성 꼬임이 적습니다.

- **최종 산출물:** 발표 영상(15분 이내) — **별도 최종 보고서 없음**
- **마감:** 영상 2026-06-21 (proposal은 제출 완료, 성적 미반영)

---

## 0. 시작 전에 — Scope를 먼저 정한다 (중요)

아이디어 문서에는 야심찬 요소가 많습니다(분리 critic, rollback, 궤적 분기점 복원, fine-tune, hidden-state 검사). 6/21까지, 그리고 rubric상 **평가/분석이 50%**(지표 10 + baseline 10 + 분석 20 + 방법 건전성 15 일부)임을 감안해, **"실제 수치 + error analysis까지 반드시 도달"**을 최우선으로 두고 기능을 2개 티어로 나눕니다.

| 티어 | 포함 | 근거 |
|------|------|------|
| **Core (반드시)** | 분리 critic(다른 계열) · 4대 조건 구조화 프롬프트 · 4-Way Gating · re-prompt 회복 · ask_user(비가역) · critical-divergence 라벨링 · 메인 비교 + 기본 ablation | 가설 검증과 rubric 충족에 필수 |
| **Stretch (시간 남으면)** | rollback-to-divergence(env checkpoint) · cross-family ablation 전체 · fine-tuned critic · hidden-state 1차 스크리닝(RepE) | 차별점 강화용. 안 되면 "future work"로 영상에 제시(분석 20%에 유리) |

> 원칙: **Core를 끝까지 돌려 수치를 확보**한 다음 Stretch에 손댄다. 미완 Stretch는 영상의 "한계 + future direction"으로 전환해 오히려 분석 점수에 활용.

---

## 전체 그림

```
Phase 0  환경·벤치마크 셋업                ─┐
Phase 1  Action taxonomy + critical-divergence 라벨링 ─┤ (직렬)
Phase 2  Critic 모듈(구조화 4-way)         ─┘
Phase 3  에이전트 통합: gating + recovery   ─┐
Phase 4  Baseline (vanilla/reflexion/saber) ─┤ (병렬 가능)
Phase 5  평가 하니스 & 지표                  ─┘
Phase 6  실험 & Ablation
Phase 7  분석 & error analysis
Phase 8  영상 제작 (rubric 기반)
```

---

## Phase 0 — 환경·벤치마크 셋업

**목표:** baseline 에이전트가 task 몇 개를 돌려 채점까지 되는 상태.

### 0.1 초기화
```bash
mkdir critic-gating && cd critic-gating
git init
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
```
### 0.2 벤치마크 submodule
```bash
git submodule add https://github.com/sierra-research/tau-bench external/tau-bench
pip install -e external/tau-bench
# τ²-bench를 쓸 경우 repo 확인 후 추가
```
### 0.3 키 & gitignore
`.env`에 `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. `.gitignore`에 `.env`, `.venv/`, `data/results/`, `__pycache__/`.

### 0.4 smoke test
```bash
python -m tau_bench.run --agent-strategy tool-calling \
  --env retail --model gpt-4o --model-provider openai \
  --user-model gpt-4o --user-model-provider openai \
  --task-split test --end-index 5
```

**✅ Done:** 5개 task가 에러 없이 돌고 성공/실패가 출력된다.
**⚠️** τ²/Verified의 도메인명·split·CLI는 클론 repo README로 확인. 안 되면 τ-bench로 폴백.

---

## Phase 1 — Action Taxonomy + Critical-Divergence 라벨링

**목표:** (a) 무엇이 mutating인지 확정, (b) **"step N에서 틀렸다"의 gold**를 만들어 Recovery Rate·Critic 지표 측정의 기반 확보. (제안서 *5월* 작업)

### 1.1 Mutating vs Read-only — `src/harness/action_taxonomy.py`
도메인별 툴을 분류표로(실제 툴명은 repo에서 확인). 예시(retail):

| 분류 | 예시 | Critic 검문 |
|------|------|-------------|
| mutating | `cancel_pending_order`, `modify_pending_order_items`, `exchange_delivered_order_items`, `book_reservation` | O |
| read-only | `get_order_details`, `find_user_id_by_*`, `list_*`, `get_product_details` | X(bypass) |

```python
MUTATING_TOOLS = {"retail": {...}, "airline": {...}}
def is_mutating(domain, tool): return tool in MUTATING_TOOLS.get(domain, set())
```
가역/비가역도 함께 표기(rollback·ask_user 분기에 필요). 예: 결제·발권 = 비가역.

### 1.2 Critical-Divergence 라벨링 — `src/eval/label_divergence.py` (핵심 난점)
τ²-bench는 **최종 DB 상태로만 채점**하고 중간 step의 옳고 그름은 알려주지 않습니다. Recovery Rate를 재려면 "어디서 처음 잘못됐나"가 필요합니다. SABER의 **critical divergence** 개념을 차용:

- 같은 task에 대해 **성공 trajectory와 실패 trajectory**를 수집한다(여러 시드/온도, 또는 baseline vs ours).
- 두 궤적을 정렬해 **상태/행동이 처음으로 갈라지는 mutating step**을 "결정적 어긋남 지점"으로 표시.
- 그 지점의 action을 "원래 Block/Revise가 맞았던 케이스"로 자동 라벨(약지도).

**2단계 라벨 전략:**
1. **자동(약지도):** 위 divergence 기반 + 최종 DB diff로 실패 유발 mutating action 마킹.
2. **수동(소량):** 도메인별 30~50개 mutating 시점을 사람이 `{approve, block, revise, ask_user}` + 가역성으로 라벨 → 4-Way Decision Accuracy의 gold.

라벨 형식(JSONL):
```json
{"task_id":"retail_012","step":7,"tool":"modify_pending_order_items",
 "args":{...},"gold_decision":"revise","reversible":true,
 "divergence":true,"reason":"수량 오류, 목표는 정상"}
```

> **🔒 누수 금지:** 이 gold는 **평가에만** 쓰고, 실행 중 Critic 프롬프트에 절대 넣지 않는다("LLMs Cannot Self-Correct… Yet"의 oracle 효과 방지).

**✅ Done:** (a) 두 도메인 mutating/가역성 목록 확정, (b) divergence 라벨러 동작, (c) 도메인별 30+ 수동 라벨, (d) 라벨 로더.

---

## Phase 2 — Critic 모듈 (구조화 4-Way)

**목표:** action 하나를 받아 4대 조건 대조 후 4-way 판정 + 회복 계획을 내리는 분리 모듈.

### 2.1 스키마 — `src/critic/schemas.py`
```python
Verdict = Literal["approve","block","revise","ask_user"]
@dataclass
class Consequence: predicted_effect: str; affected_state: str
@dataclass
class Decision:
    verdict: Verdict; reason: str
    revised_args: Optional[dict] = None       # revise
    question_to_user: Optional[str] = None    # ask_user
    reversible: Optional[bool] = None          # 회복 전략 선택용
    rollback_to_step: Optional[int] = None     # (stretch) 복원 지점
```

### 2.2 구조화 프롬프트 — `src/critic/prompts.py`
4대 조건을 **명시적으로** 나열하고 **JSON으로만** 답하게 강제:
```
[역할] 환경 변경 직전 action을 검문하는 Critic.
[입력] 유저 목표 · 현재 상태/대화이력 · 도메인 정책 · 제안 action(tool+args)
[1] 예상 결과(consequence) 1문장 추론
[2] 4대 조건 각각 부합 여부 + 근거: User Goal / State / Constraints / Domain Policy
[3] 가역성 판단
[4] 최종 4-way: approve | block | revise(+args) | ask_user(+question)
[출력] 지정 JSON 스키마만. 확실치 않으면 approve로 치우치지 말고 ask_user.
```
> 도메인 정책 전문을 프롬프트에 넣는 게 Domain Policy 대조의 핵심 — ablation에서 조건별로 빼며 기여도 측정.

### 2.3 호출·파싱 — `src/critic/critic.py`
- **분리 모델 사용:** 본체와 다른 계열/인스턴스를 `CRITIC_MODEL`로. (Core 차별점)
- JSON 파싱 실패/타임아웃 fallback 정책 명시(예: `ask_user`).
- (Stretch) `src/critic/repe.py`: 본체 hidden-state로 1차 위험도 스코어 → 임계 초과 시에만 full Critic 호출(비용 절감 + RepE 차별점).

**✅ Done:** 임의 입력에 항상 유효한 `Decision` 반환. 단위 테스트로 4가지 verdict 케이스 확보.

---

## Phase 3 — 에이전트 통합: Gating + Recovery

**목표:** 실제 루프에서 게이팅과 회복이 동작.

### 3.1 래퍼 — `src/agents/critic_agent.py`
```python
proposed = agent.next_action(obs)
if not is_mutating(domain, proposed.tool):
    result = env.step(proposed)                      # read-only 통과
else:
    if STRETCH_ROLLBACK: ckpt = snapshot(env)        # mutating 전 스냅샷
    d = critique(goal, state, policy, proposed)       # 분리 critic
    if   d.verdict=="approve":  result = env.step(proposed)
    elif d.verdict=="revise":   result = feedback("REVISE: "+d.reason)  # 또는 d.revised_args 재시도
    elif d.verdict=="ask_user": result = ask_user(d.question_to_user)   # 비가역 우선
    elif d.verdict=="block":
        if d.reversible and STRETCH_ROLLBACK:
            restore(env, d.rollback_to_step or ckpt)  # 분기점으로 복원
        result = feedback("BLOCKED: "+d.reason)        # 사유·피드백 주입(재발 방지)
```
- **재발 방지:** block/revise 시 사유와 이전 실수 요약을 다음 관측에 주입.
- **무한 루프 방지:** 동일 action 재시도 상한(예: 2회) 초과 시 `ask_user` 또는 종료.

### 3.2 Recovery 모듈 (Stretch)
- `recovery/checkpoint.py`: env 상태 deepcopy 스냅샷/복원.
- `recovery/divergence.py`: Critic이 지목한, 혹은 휴리스틱으로 추정한 분기 step으로 복원.

### 3.3 로깅
매 step `{tool,args,is_mutating,reversible,decision,executed?,rolled_back?}` 기록 — 지표의 원천.

**✅ Done:** task 1개를 critic_agent로 끝까지 실행 시, mutating step마다 4-way 판정이 로그에 남고 block/revise가 실제 실행을 막는다. (Stretch면 rollback 후 정상 진행 확인)

---

## Phase 4 — Baseline

동일 하니스 위에서 공정 비교:
- **Vanilla:** 보호장치 없는 tool-calling 에이전트.
- **Reflexion:** task 실패 후 같은 모델이 반성문을 다음 시도에 주입(사후·동일모델).
- **SABER:** mutating 직전 같은 모델로 단순 prompting 검증(중간·동일모델·비구조화·rollback 없음).

> 풀 재현이 어려우면 **핵심 메커니즘만 충실히 재현한 간소화 버전**으로 두고 영상/문서에 명시(시간 예산상 합리적). 단, "사후 vs 사전", "동일모델 vs 분리"라는 비교 축은 반드시 보존.

**✅ Done:** 세 baseline 모두 runner로 동일 호출 가능.

---

## Phase 5 — 평가 하니스 & 지표

### 5.1 Runner — `src/harness/runner.py`
`configs/experiment.yaml`(method × domain × model × seeds × split) 순회 실행, `data/results/`에 trajectory·요약 저장.

### 5.2 지표 — `src/eval/metrics.py`

| 그룹 | 지표 | 정의 |
|------|------|------|
| Task | pass@1 | 1회 시도 성공률 |
| Task | pass^k | k회 반복 모두 성공한 비율(일관성) |
| Task | **Recovery Rate** | divergence 발생 task 중 최종 성공 비율 — **가설 직접 검증** |
| Critic | Precision/Recall | gold(차단이 맞았던 케이스) 대비 block 판정 정확도 |
| Critic | False Block Rate | approve여야 할 action을 block한 비율 |
| Critic | 4-Way Decision Accuracy | 수동 라벨 대비 4-way 일치율 |
| Critic | Reversibility Accuracy | 가역성 판단 정확도(오판 = 치명적 잔존오류) |
| 실용성 | Latency / Token Cost | Critic 개입으로 늘어난 task당 평균 |

> **개발 절약:** full run은 비싸다(에이전트+유저시뮬+critic 3종 호출). 개발 중엔 `--end-index 5~10`으로 파이프라인만 검증, 정식 측정은 마지막에 도메인 전체로. 시드 고정·결과 캐싱.

**✅ Done:** `run_experiment.sh` 한 번으로 모든 method의 위 지표가 csv/json으로 산출.

---

## Phase 6 — 실험 & Ablation

**6.1 메인 비교:** 두 도메인 × {vanilla, reflexion, saber, ours} 전체 측정.

**6.2 Ablation (rubric의 분석 점수 직접 겨냥):**
- **구조화 효과:** basic critic(단일 질문) vs structured(4대 조건). 추가로 −Goal/−State/−Constraints/−Policy 조건 제거로 **어떤 조건이 탐지에 가장 기여**하는지 분해.
- **Gating 세분화:** 4-way vs 2-way(approve/block만) — revise/ask_user의 가치.
- **분리 효과(핵심 주장 검증):** 동일 계열(Claude+Claude) vs 다른 계열(Claude+GPT) vs (Stretch)fine-tuned critic.
  - → **같은 계열도 효과가 비슷하면** "분리"가 아니라 "그냥 두 번 본 효과"라는 반론이 성립. 이 ablation 없이는 분리 주장이 약함.
- **(Stretch) rollback 유무:** re-prompt만 vs rollback-to-divergence.

**✅ Done:** 메인 비교표 + ablation 표 완성, 가설 지지/반증 그림 확보.

---

## Phase 7 — 분석 & Error Analysis (rubric 20%, 최대 비중)

- **정량:** 성공률·recovery 막대, FBR vs Recall trade-off, latency/cost 대비 이득.
- **정성(error analysis):** Critic이 ① 잘 잡은 케이스, ② **오차단(false block)**, ③ **놓친 케이스**, ④ 가역성 오판으로 인한 치명적 잔존오류 사례를 trajectory와 함께 정리.
- **한계:** Critic 비용, fallback 민감도, 도메인 일반화, oracle 없는 자기수정의 본질적 한계.
- **Future direction:** 미완 Stretch(fine-tune critic, RepE 1차 검사, 분기점 복원 정교화)를 논리적 다음 단계로 제시.

**✅ Done:** 강점·한계·error analysis·future가 슬라이드/노트로 정리.

---

## Phase 8 — 영상 제작 (≤15분, rubric 기반)

rubric을 그대로 목차로 삼아 시간을 배분(괄호=배점):

| 섹션 | 분량(목표) | rubric |
|------|-----------|--------|
| Problem & Motivation | ~1.5분 | Problem Definition 10% |
| Related Work & 위치짓기 | ~1.5분 | Related Work 10% |
| Method (4대 조건·4-way·분리·회복) | ~3분 | Novelty 10% + Soundness 15% |
| Eval 설계(지표·baseline·divergence 라벨) | ~2분 | Metrics 10% + Baselines 10% |
| 결과 + **Ablation + Error Analysis + Future** | ~5분 | **Analysis 20%** |
| 마무리 | ~1분 | Clarity 10% |

- **Length 5%:** 15분 절대 초과 금지(타이머 확인).
- 분석 섹션을 가장 길게. 미완 부분은 "한계+future"로 전환해 손해를 점수로 바꾼다.

**✅ Done:** 15분 이내 영상 + 업로드 폴더 제출.

---

## 타임라인 (~6/21)

| 시기 | Phase | 산출물 |
|------|-------|--------|
| 5월 | 0, 1 | 셋업·smoke test, action/가역성 taxonomy, divergence 라벨러, 수동 30+/도메인 |
| 6월 초 | 2, 3 | 구조화 4-way Critic, gating+recovery 루프 |
| 6월 중 | 4, 5 | baseline 3종, 평가 하니스·지표 |
| 6월 중후반 | 6 | 메인 비교 + ablation(특히 분리/구조화) |
| ~6/18 | 7 | 분석·error analysis·시각화 |
| ~6/21 | 8 | 영상 제작·제출 |

---

## 먼저 합의할 의사결정

1. **벤치마크:** τ-bench(안정) vs τ²-bench Verified(제안서). dual-control이 핵심인지로 결정.
2. **모델 구성:** 에이전트 / 유저시뮬 / Critic 각각 모델. Critic을 다른 계열로(분리 주장 위해) + 비용 고려.
3. **Critic fallback:** 불확실 시 `approve`(비개입) vs `ask_user`(안전). 권장: `ask_user`.
4. **재시도 상한 / rollback 채택 여부**(Stretch).
5. **Baseline 재현 수준:** 풀 vs 간소화 — 시간 기준으로 결정하고 영상에 명시.
6. **fine-tune critic / RepE 1차 검사** 시도 여부(Stretch, 데이터·시간 예산 확인).

---

## 리스크 & 대응

| 리스크 | 대응 |
|--------|------|
| 토큰/시간 비용 폭증 | 소규모 split 개발, mutating에만 critic 호출, (stretch)RepE 사전 스크리닝, 캐싱 |
| τ² repo/CLI 불일치 | 클론 후 README 확인, 안 되면 τ-bench 폴백 |
| Recovery Rate 라벨 어려움 | critical-divergence 약지도 + 소량 수동, 신뢰구간 보고 |
| 정상 action 과차단(FBR↑) | FBR을 1급 지표로, 프롬프트에 "불확실시 ask_user" |
| 분리 주장 반론(두 번 본 효과) | cross-family ablation 필수 수행 |
| 비가역 action 가역성 오판 | reversibility accuracy 추적, 비가역 default = ask_user |
| Stretch 미완 | future work로 전환, 분석 점수로 활용 |
| 영상 15분 초과 | 분석 우선 배분, 사전 리허설·타이머 |

---

## Deliverables 체크리스트

- [ ] `external/tau-bench` 셋업 + smoke test
- [ ] `action_taxonomy.py` (mutating/가역성, 두 도메인)
- [ ] `label_divergence.py` + 수동 30+/도메인 라벨
- [ ] `critic/{schemas,prompts,critic}.py` (+선택 repe.py) + 단위테스트
- [ ] `critic_agent.py` gating(+recovery) + 로깅
- [ ] baseline 3종 (vanilla/reflexion/saber)
- [ ] `runner.py` + `metrics.py` + `run_experiment.sh`
- [ ] 메인 비교표 + ablation(구조화/4-way/분리)
- [ ] error analysis + 시각화
- [ ] **발표 영상(≤15분) 제출**
