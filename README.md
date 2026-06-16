# Goal-State-Constraint Aware Critics for Pre-Action Error Detection & Recovery in Long-Horizon LLM Agents

**Group 18**

Long-horizon agent task에서 초반 오류가 누적되어 전체가 실패하는 문제(compounding error)를, 메인 agent와 **분리된 Critic LLM**으로 막는다. Critic은 환경을 바꾸는 행동(mutating action) 직전에 예상 결과를 4대 조건(목표·상태·제약·정책)과 대조해 4-way로 판정하고, 막은 경우 재프롬프트·궤적 복원·사용자 확인으로 회복한다.

---

## 1. 문제와 가설

- **문제:** 20+ step task에서 초반 오류 하나가 이후 단계 전부로 전파된다. tau-bench에서 SOTA도 pass@1 50~60%, pass^k 25% 미만 — 한 번은 풀어도 일관되게는 못 푼다.
- **재해석:** 실패는 "잘 푸는 능력의 부족"이 아니라 "틀린 것을 빨리 알아채고 회복하는 능력의 부족" 때문이다.
- **가설:** 분리된 Critic으로 환경 변경 직전에 게이팅하면, 같은 모델이 사후에 점검하는 방식(Reflexion)보다 **회복률(recovery rate)이 높다.**

---

## 2. 기존 연구와의 차별점

| 연구 | 개입 시점 | 검증 주체 | 회복 |
|------|-----------|-----------|------|
| Reflexion (2023) | 사후 | 동일 모델 | 다음 시도에 반성문 주입 |
| SABER (2025) | mutating 직전 | 동일 모델, 단순 prompting | rollback 없음 |
| **Ours** | mutating 직전 | **분리된 Critic**(별도 모델) | 재프롬프트 / 분기점 복원 / 사용자 확인 |

개입 시점(mutating action 직전)은 SABER에서 가져온다. 새로움은 세 가지다.
1. **4대 조건 구조적 대조 + 4-Way Gating** (비구조화 prompting이 아님)
2. **분리된 Critic** (main agent와 다른 별도 모델 — `gpt-5.4-mini` vs `gpt-5.4-nano`)
3. **분기점 복원 회복** (직전 단계가 아니라 처음 어긋난 지점으로 복원)

> Critic은 평가 중 정답 라벨을 절대 보지 않는다. 이득이 "정답을 본 효과"가 아님을 ablation으로 입증한다.

---

## 3. 핵심 메커니즘

조회 행동(`get_*`, `find_*`)은 통과시키고, 환경을 바꾸는 행동(mutating action) 직전에만 Critic이 개입한다.

```
Main agent ──제안──▶ [mutating action인가?]
                          │ 아니오 ─────────────────▶ 바로 실행
                          │ 예
                          ▼
                    Critic LLM (분리된 모델)
                    1) 예상 결과 추론
                    2) 4대 조건 대조 (목표 / 상태 / 제약 / 정책)
                    3) 가역성 판단
                          ▼
              ┌──── 4-Way Gating ────┐
              │ Approve  → 실행        │
              │ Revise   → 인자 고쳐 재시도 │
              │ Block    → 차단 + 회복 │
              │ Ask_user → 사용자 확인 │
              └───────────────────────┘
```

**판정 규칙**
- **Approve:** 목표와 맞고 제약·정책 위반 없고 도구·인자 모두 옳음.
- **Revise:** 도구·의도·대상은 옳고 인자 값만 틀려 국소 수정 가능.
- **Block:** 정책·상태를 명백히 어긴 행동 → **가역성과 무관하게** 거부 + 피드백(가역이면 분기점 복원). 위반은 사용자에게 승인을 묻지 않는다.
- **Ask_user:** 정보가 모자라거나, **유효하지만** 되돌릴 수 없는 중대 행동(사용자 확인 필요).

가역성은 *명백한 위반*을 가르지 않는다(위반은 항상 block). 가역성은 *유효하지만 위험한* 행동에서 **진행 vs 확인(ask_user)**을 가르고, 잘못된 block을 가역 오판으로 강행하면 치명적 잔존 오류가 남으므로 가역성 판단 정확도가 여전히 중요하다.

---

## 4. 벤치마크 — tau-bench

tau-bench는 정적 데이터셋이 아니라 **데이터 + 실행 환경 + 정답**이 든 실행형 패키지다(`sierra-research/tau-bench`). 구성: 가짜 DB(JSON), 그 DB를 조작하는 로컬 도구 함수, task 정의(고객 요청 · 정답 액션 목록 · 정답 최종 상태), LLM 사용자 시뮬레이터. 외부 결제·항공사 연동은 없고 LLM API 키만 필요하다.

- **주력:** retail. **선택(stretch):** airline, tau²-bench Verified.

---

## 5. 데이터 — 직접 만드는 라벨 (SFT 여부와 무관하게 필수)

tau-bench는 정답 액션은 주지만 "몇 번째가 어떻게 틀렸는가"는 안 준다. 회복률 측정과 Critic 정확도 측정에 이 라벨이 필요하므로 직접 만든다.

1. **정답 액션 변형(주력):** 정답 액션을 일부러 망가뜨려 라벨된 오답 생성. 인자만 틀리면 `revise`, 정책 위반이면 `block`, 정보 부족이면 `ask_user`, 그대로면 `approve`. agent 실행 불필요.
2. **분기점 마이닝(보완):** 같은 task의 성공·실패 로그를 비교해 처음 갈라지는 단계를 표시. 회복률 측정의 기준 라벨.

> 이 라벨은 평가·학습에만 쓰고 실행 중 Critic 프롬프트에 넣지 않는다.

---

## 6. 평가

retail(+가능하면 airline)에서 `vanilla`, `Reflexion`, `SABER`, `Ours`, `Oracle critic(상한)`을 동일 조건으로 비교한다.

- **Task:** pass@1, pass^k, **Recovery Rate(핵심)**
- **Critic:** Precision/Recall, False Block Rate, 4-Way Accuracy, 가역성 정확도
- **실용성:** Latency / Token Cost
- **Ablation:** ① 구조화 vs 단순 critic, ② 4-way vs 2-way, ③ 동일 계열 vs 다른 계열 critic

---

## 7. 역할 분담 (4인)

| 담당 | 영역 |
|------|------|
| **P1 (나)** | 파이프라인 + Critic + baseline(vanilla/Reflexion/SABER) + oracle critic |
| **P2 (데이터)** | 도구·가역성 분류표, 정답 액션 변형 생성기, 분기점 라벨러 |
| **P3 (실험)** | metric 코드, 반복 실행 runner, 정량 평가·숫자 분석 |
| **P4 (영상)** | 발표 영상 + (빌드 기간) 결과 시각화·error analysis 사례 정리 |

- 파이프라인(P1)은 데이터(P2)를 기다리지 않는다 — P1의 프롬프트 critic으로 루프를 먼저 돌린다.
- 분기점 라벨러(P2)는 P3가 돌린 실제 로그에 적용한다(손발 맞춤).
- 상세 실행 순서·산출물은 `ACTION_PLAN.md` 참조.
- 구현 코드: `pipeline_code/`. 실행 방법: P1 = `implementation_detail_p1.md`, P2(데이터·라벨·평가) = `implementation_detail_p2.md`.
- P2↔P1/P3 합의 필요 사항: `integration_issues_p2.md`.

**모델 구성 (구현):** Main Agent = `gpt-5.4-nano`, Critic (Ours) = `gpt-5.4-mini`, API 키 1개(`OPENAI_API_KEY`)만 필요.

---

## 8. References

- Barres et al. (2025). τ²-Bench. arXiv:2506.07982.
- Cuadron et al. (2025). SABER. arXiv:2512.07850.
- Ghasemabadi & Niu (2025). Can LLMs Predict Their Own Failures? arXiv:2512.20578.
- Huang et al. (2024). LLMs Cannot Self-Correct Reasoning Yet. arXiv:2310.01798.
- Oh et al. (2026). Uncertainty Quantification in LLM Agents. arXiv:2602.05073.
- Shinn et al. (2023). Reflexion. arXiv:2303.11366.
- Yao et al. (2024). τ-Bench. arXiv:2406.12045.
