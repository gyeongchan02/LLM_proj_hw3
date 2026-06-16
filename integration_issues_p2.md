# P2 ↔ P1/P3 합의 필요 사항 (Consensus Items)

P2(데이터/라벨)가 P1(critic·oracle)·P3(metrics)와 맞물리는 지점에서, **팀 합의가
필요한** 정합성 사안만 정리한다. (P2 단독으로 해결된 항목은 본 문서에서 제외 — 하단
"P2 측 완료" 참조.)

검증: conda `llm_hw3`, tau-bench retail, 도구 직접 호출 + 코드 추적으로 확인.

---

## 배경 — 핵심 구분

`perturbations.jsonl`의 각 행은 *"(특정) 제안 행동 → 올바른 verdict"* 쌍이다. 따라서
**행동 단위(offline)** 로 소비해야 한다. P2가 그 올바른 소비 경로를
`src/eval/critic_accuracy.py`(오프라인 critic-정확도 하니스)로 구현했다 — 각 라벨의 행동을
critic에 직접 먹여 verdict를 gold와 비교(에이전트 롤아웃 불필요, critic LLM 호출만).

이로써 **Critic 정확도 지표(4-way/precision/recall/false-block/reversibility)는 라이브
실행과 무관하게 올바르게 측정 가능**해졌다. 아래는 그래도 남는 합의 사항이다.

---

## C1 — Critic 정확도 측정 경로 (P3)

**결정 필요.** Critic 정확도(README §6 "Critic" 그룹)를 **오프라인 하니스로 측정**할지.

- P2 제공: `src/eval/critic_accuracy.py` — 라벨 행동을 critic에 직접 투입, 4-way/precision/
  recall/false-block/reversibility + 조건별(GOAL/STATE/CONSTRAINT/POLICY)·basis별 분해.
  ours/saber critic 비교, `--ablate <조건>` 지원.
- `src/eval/metrics.py::compute_critic_metrics`는 **라이브 실행 로그**를 `(task_index, tool)`로
  라벨과 매칭하는데, 이는 잘못이다: 라이브에서 에이전트는 *올바른* 행동을 제안하므로, 그
  approve 판정을 *망가진* 행동용 `gold_decision`과 비교하게 된다(서로 다른 행동 비교).
  또 같은 도구가 한 task에 2~3회 나오는 **29개 task**에서 2·3번째 호출이 첫 호출 라벨로
  오채점된다.

**권고.** Critic 정확도는 오프라인 하니스(`critic_accuracy.py`)로 산출. `compute_critic_metrics`의
라이브 (task,tool) 매칭은 **deprecate 또는 비활성화**(혼동 방지). 라이브 실행은 Task 지표
(pass@1/pass^k/Recovery/비용)에만 사용.

---

## C2 — Oracle "천장" 재설계 (P1)

**결정 필요.** 현재 oracle 천장이 제 역할을 못 한다.

`src/critic/critic.py::critique_oracle`는 `(task_index, tool)`로만 라벨을 조회하고 `tool_args`를
무시한다. 라이브 oracle 실행에서 에이전트는 *올바른* gold 행동을 제안하는데, oracle은 그
(task,tool)의 perturbation verdict(대개 비-approve)를 반환 → **올바른 행동을 차단**한다.
(재현: task 4 `modify_pending_order_items`에 올바른 gold 인자를 줘도 oracle = `block`.)
결과적으로 "완벽 탐지기 천장"이 vanilla보다 낮아질 수 있다.

**선택지.**
- (a) oracle을 "에이전트 제안이 gold 행동에서 벗어났는지" 탐지하는 방식으로 재설계
  (gold와 일치하면 approve, 벗어나면 그 일탈에 맞는 verdict).
- (b) 라이브 oracle 천장을 포기하고, oracle 의미를 "오프라인 하니스에서 gold 라벨을 그대로
  반환하는 상한"으로 한정(= 하니스 4-way accuracy의 이론적 1.0 상한).

**권고.** (a)가 README의 "perfect detector" 취지에 맞다. P1이 oracle/gated 경로를 정해야 함.

---

## C3 — Recovery Rate용 분기 라벨 (P3 ↔ P2)

**결정 필요(이미 설계된 공동 단계).** 현재 `divergences.jsonl`은 **bootstrap**
(perturbation의 non-approve task)이라, *실행에서 실제 어긋난* 집합이 아니다. 이걸
`compute_recovery_rate`에 먹이면 "P2가 고른 부분집합의 성공률"이 되어 자기참조가 된다.

**해결 경로(완성된 도구 사용).** P3가 실험 로그를 만들면, P2의 `label_divergence.py`를
`vs-gold`(실행 vs 정답) 또는 `pairwise`(성공 vs 실패) 모드로 돌려 **진짜 분기점**으로 재생성.
즉시 가능한 보강: `compute_recovery_rate`가 bootstrap 파일을 받으면 경고/거부하는 가드.

---

## P2 측 완료 (참고)

- `action_taxonomy.py`: 실제 16개 도구와 정확 일치(런타임 `get_info`), 가역성 재분류
  (cancel/return/modify_items=irreversible, 근거 주석), 정책 텍스트=실제 `wiki.md`.
- `perturb.py`: 10개 연산자(4조건 커버)·실제 env 실증·basis/evidence 부착.
  - foreign_order가 task마다 다른 *진짜 남의 주문* 사용(이전: 전부 동일 주문 — 수정됨).
  - approve 라벨의 gold 행동이 env에서 실행되는지 검증, 실패하는 gold 4개 제외.
  - GOAL 라벨 20개로 보강(조건 ablation 표본 확보).
- `label_divergence.py`: vs-gold/pairwise/bootstrap 3모드(vs-gold 합성 로그 검증).
- `critic_accuracy.py`: 오프라인 critic-정확도 하니스(mock critic으로 배관 검증).
- (확인) corrupt_item "+9"의 실재 ID 충돌 우려는 **오판** — 모든 변형 ID가 10자리,
  "+9"는 11자리라 충돌 불가. 변경 불필요.

현재 라벨: `perturbations.jsonl` 146개, `divergences.jsonl`(bootstrap, C3에서 재생성 예정).
