# Goal-State-Constraint Aware Critics for Pre-Action Error Detection & Recovery in Long-Horizon LLM Agents

**Group 18** 

Long-horizon agent task에서 초기 오류가 누적되어 전체 궤적이 실패하는 문제(compounding error)를, **메인 에이전트와 분리된 Critic LLM**으로 막는 연구입니다. Critic은 환경을 바꾸는 action(mutating action) 직전에 예상 결과를 4대 조건(목표·상태·제약·정책)과 구조적으로 대조해 4-way로 게이팅하고, 막은 경우에는 단순 재프롬프트·궤적 복원(rollback)·사용자 확인으로 회복을 유도합니다.

---

## 1. 무엇을 푸는가

- **문제 현상:** 20+ step long-horizon task에서, 초반의 작은 잘못된 action이 이후 궤적 전체를 오염시킴(compounding error).
- **경험적 근거:** τ-bench에서 SOTA 모델도 성공률이 절반 수준이고, 같은 task를 여러 번 시키면 매번 성공하는 비율(pass^k)은 1/4 미만 — 즉 *한 번은 풀어도 일관되게는 못 푼다*.
- **핵심 재해석:** long-horizon 실패는 **"잘 푸는 능력 부족"이 아니라 "틀린 걸 빨리 알아채고 회복하는 능력 부족"** 때문이다.
- **문제 재정의:** 에이전트 오류 복구를 **"Pre-action Error Detection & Gating(+Recovery)"** 문제로 구체화하고, mutating action 직전에만 개입하는 **별도** Critic을 둔다.

### 가설
> 본체 에이전트와 분리된 Critic LLM을 두고 환경 변경 직전에 검증·게이팅하면, **같은 모델이 자기 자신을 사후에 돌아보는 방식(Reflexion 등)보다 회복률(recovery rate)이 높아진다.**

---

## 2. 기존 연구와의 차별점

| 연구 | 개입 시점 | 검증 주체 | 회복 | 한계 |
|------|-----------|-----------|------|------|
| Reflexion (2023) | 사후(task 종료 후) | 본체와 동일 모델 | 다음 시도에 반성문 주입 | 진행 중 차단 불가, 같은 모델→같은 실수 반복 위험 |
| SABER (2025) | mutating action 직전 | 본체와 동일 모델, 단순 prompting | rollback 없음, reflection 주입만 | 검토 항목 비구조화, 분리된 검증자 아님 |
| **Ours** | mutating action 직전 | **분리된 Critic**(다른 계열, 선택적 fine-tune) | re-prompt / 분기점 복원 / 사용자 확인 | — |

본 연구의 차별점은 **(1) 4대 조건 명시적 대조 + 4-Way Gating**, **(2) 분리된 Critic**(동일 계열이 아니라 별도 모델/특화), **(3) "어디서 꼬였는지" 짚어 그 지점으로 복원하는 회복 전략**입니다.

> **주의(방법론적 함정):** "LLMs Cannot Self-Correct Reasoning Yet"는 *외부 정답(oracle) 없이* 모델이 스스로 유지/수정을 판단하면 오히려 정확도가 떨어질 수 있음을 보였습니다. 따라서 Critic이 평가 중 gold label을 절대 보지 않도록 설계해야 하며(데이터 누수 금지), Critic의 이득이 "정답을 본 효과"가 아님을 ablation으로 입증해야 합니다.

---

## 3. 핵심 메커니즘

에이전트가 action을 제안하면 그것이 **mutating action**(결제·예약·DB 변경 등 환경을 바꾸는 write API)인지 확인합니다. read-only(`get_*`/`find_*` 등 조회)는 Critic을 거치지 않고 바로 실행됩니다. SABER의 관찰("결정적 오류는 mutating step에서만 발생")을 따라 **검증 비용을 결정적 시점에 집중**합니다.

```
Main agent ──proposes──▶ [mutating action?]
                              │ no ─────────────────────────────▶ Environment (바로 실행)
                              │ yes
                              ▼
                        Critic LLM  (분리된 모델)
                        ├─ (선택) hidden-state 1차 스크리닝
                        ├─ 1) 예상 결과(consequence) 추론
                        └─ 2) 4대 조건 대조
                              · User Goal     (유저가 원한 것과 맞나)
                              · State         (현재 DB/대화 상태와 모순 없나)
                              · Constraints   (수량/시점/권한 등 제약 위반 없나)
                              · Domain Policy (도메인 규칙 위반 없나)
                              ▼
                  ┌──── 4-Way Gating ───────────────────────────┐
                  │ Approve   → 그대로 실행                       │
                  │ Block     → 차단 + 회복 전략                  │
                  │ Revise    → argument만 고쳐 재시도            │
                  │ Ask_user  → (특히 비가역) 유저에게 확인        │
                  └──────────────────────────────────────────────┘
                              │  Block/Revise의 회복 전략:
                              │   · re-prompt: 사유·피드백 주입해 같은 실수 반복 방지
                              │   · rollback: 직전 step이 아니라 "꼬인 분기점"으로 복원
                              └─▶ 다시 에이전트 루프로
```

회복 전략 두 축:
- **가역 action** → rollback. 단순 직전 복귀가 아니라 Critic이 **궤적이 어긋난 지점**을 역추적해 그곳으로 되돌리고, 같은 실수를 막도록 피드백을 주입.
- **비가역 action**(실제 결제 등) → 실행 전 **사용자에게 확인**해 우회. Critic의 가역성 오판이 곧 치명적 잔존 오류이므로 가역성 판단 정확도가 중요.

---

## 4. 레포 구조

```
critic-gating/
├── README.md                  # 이 문서
├── ACTION_PLAN.md             # 상세 end-to-end 실행 계획 (rubric 매핑 포함)
├── .env / .gitignore / requirements.txt
├── external/
│   └── tau-bench/             # 벤치마크 (git submodule, 수정 X · import만)
├── src/
│   ├── critic/
│   │   ├── schemas.py         # Consequence, Decision, RecoveryPlan dataclass
│   │   ├── prompts.py         # 4대 조건 구조화 프롬프트
│   │   ├── critic.py          # Critic LLM 호출 + 4-way gating
│   │   └── repe.py            # (선택) hidden-state 1차 스크리닝
│   ├── recovery/
│   │   ├── checkpoint.py      # 환경 state 스냅샷/복원 (rollback 기반)
│   │   └── divergence.py      # 궤적 분기점 추정 + 복원 지점 선택
│   ├── agents/
│   │   ├── critic_agent.py    # 메인: agent loop + gating + recovery
│   │   ├── reflexion_agent.py # baseline
│   │   └── saber_agent.py     # baseline
│   ├── harness/
│   │   ├── action_taxonomy.py # 도메인별 mutating vs read-only 분류
│   │   └── runner.py          # task 실행 + trajectory 로깅
│   └── eval/
│       ├── label_divergence.py# 성공/실패 궤적 비교로 "결정적 어긋남" gold 생성
│       ├── metrics.py         # success / recovery / precision-recall / FBR
│       └── analyze.py         # 집계·ablation·error analysis
├── configs/experiment.yaml
├── data/{annotations,results}/
└── scripts/{setup.sh,run_experiment.sh}
```

> **설계 의도:** 벤치마크는 수정 없이 submodule로 두고 import만 합니다. Critic은 "action 제안 시점"과 "환경 실행 시점" 사이에 끼어들어야 하므로 `critic_agent.py`가 기존 루프를 **감싸서(wrap)** hook을 겁니다. rollback을 위해 `recovery/checkpoint.py`가 매 mutating step 전 환경 상태를 스냅샷합니다.

---

## 5. Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install -U pip
git submodule add https://github.com/sierra-research/tau-bench external/tau-bench
pip install -e external/tau-bench
# .env 에 ANTHROPIC_API_KEY / OPENAI_API_KEY 설정

# baseline smoke test (task 5개)
python -m tau_bench.run --agent-strategy tool-calling \
  --env retail --model gpt-4o --model-provider openai \
  --user-model gpt-4o --user-model-provider openai \
  --task-split test --end-index 5
```

> ⚠️ τ²-bench 및 "Verified" 서브셋은 최신 자료라 repo 경로/CLI가 다를 수 있습니다. 클론 직후 README로 도메인·split·실행 명령을 확인하세요. τ-bench 본체(`sierra-research/tau-bench`, retail/airline)는 안정적입니다.

---

## 6. 평가 개요

τ²-bench Verified의 쇼핑·항공 task에서 `기본 agent`, `Reflexion`, `SABER`, `Ours`를 동일 조건으로 비교합니다.

- **Task-level:** pass@1, Consistent Success(pass^k), **Recovery Rate(핵심 — 가설 직접 검증)**
- **Critic-level:** Precision/Recall, False Block Rate, 4-Way Decision Accuracy, 가역성 판단 정확도
- **실용성:** Critic 개입으로 늘어난 Latency / Token Cost
- **Ablation:** ① 구조화 Critic vs basic, ② 4-way vs 2-way, ③ 동일 계열(Claude+Claude) vs 다른 계열(Claude+GPT) vs (선택)fine-tuned Critic

> **측정의 난점:** τ²-bench는 최종 DB 상태로만 채점하고 중간 step의 옳고 그름은 알려주지 않습니다. Recovery Rate를 재려면 "step N에서 틀렸다"를 라벨링해야 하므로, SABER의 **"결정적 어긋남(critical divergence)"**(성공/실패 궤적을 비교해 처음 갈라지는 지점) 개념을 빌려 gold를 만듭니다(`eval/label_divergence.py`).


---

## 7. References

- Barres et al. (2025). τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982.
- Cuadron et al. (2025). SABER: Small Actions, Big Errors — Safeguarding Mutating Steps in LLM Agents. arXiv:2512.07850.
- Ghasemabadi & Niu (2025). Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits. arXiv:2512.20578.
- Huang et al. (2024). Large Language Models Cannot Self-Correct Reasoning Yet. arXiv:2310.01798.
- Oh et al. (2026). Uncertainty Quantification in LLM Agents. arXiv:2602.05073.
- Shinn et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366 / NeurIPS 36.
- Yao et al. (2024). τ-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045.
