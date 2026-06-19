# SABER fidelity — paper vs implementation, and the faithful `saber_paper` method

**작성자:** P3 · **날짜:** 2026-06-18 · **논문:** Cuadron et al. (2025), *SABER: Small Actions, Big Errors* (arXiv:2512.07850)

## 1. 실제 SABER (논문 §4)

SABER = main model + **auxiliary model**(기본 동일 모델)로 돌아가는 test-time safeguard. 세 메커니즘:

1. **Mutation-Gated *User* Verification** — mutating 행동만 게이팅. auxiliary가 tool call을
   자연어 요약으로 바꿔 **사용자(시뮬레이터)에게 확인 요청**. 사용자 응답이 trajectory에 들어가고,
   **main model**이 실행(확인 시) 또는 수정(거부 시). 논문: "the next post-feedback action is
   executed directly" → stalling 방지.
2. **Targeted Reflection** — mutation 시점에 핵심 정책 제약 1줄 reminder 주입(`<think>` 또는 ReAct-style).
3. **Block-based Context Cleaning** — trajectory를 블록으로 나눠 임베딩·top-N(=16) 검색으로 컨텍스트 정리.

판정 주체는 **사용자**(자율 LLM veto 아님). 기본은 main=auxiliary 동일 모델, mixed pairing도 실험.

## 2. 기존 팀 `saber`가 틀린 점

팀의 `saber_agent.py`는 **자율 4-way LLM critic**(approve/block/revise/ask_user, 단순 프롬프트, 동일 모델)이다.
실제 SABER와 비교하면:

| 항목 | 논문 SABER | 팀 `saber` | 일치 |
|------|-----------|-----------|:---:|
| mutating에서만 게이팅 / 30턴 / 동일모델 | ✅ | ✅ | ✅ |
| **판정 주체** | 사용자 | LLM 자율 | ❌ |
| **사용자 시뮬레이터로 라우팅** | ✅(핵심) | ❌(에이전트에 메시지) | ❌ |
| Targeted Reflection | ✅ | ❌ | ❌ |
| Context Cleaning | ✅ | ❌ | ❌ |
| stalling 방지 | ✅ | ❌(차단 루프 유발) | ❌ |

즉 팀 `saber`는 SABER의 정의적 메커니즘(사용자 확인)을 **자율 차단**으로 바꿔, SABER가 막으려던
over-blocking/stalling을 오히려 재도입했다. (역설: 팀의 `ours`가 정책 주입 측면에서 SABER의
Targeted Reflection에 더 가깝다.)

> **명명(현재):** 충실 재현 = 메서드 **`saber`** (이제 이게 THE SABER). 원래 단순 버전 = **`saber_old`** (Old SABER, 참고용 보존).

## 3. 새 메서드 `saber` (충실 SABER) — core 2

비용·복잡도를 고려해 **gain을 만드는 두 메커니즘**(논문 Table 3)만 충실 구현. Context Cleaning(메커니즘 3)은
별도 에이전트 루프 + 턴마다 임베딩 검색이 필요해 범위에서 제외(문서화).

- **Mechanism 1 (user-gated verification):** mutating 행동 직전, `saber_verify()`(auxiliary)가
  reminder + 사용자용 확인 질문 생성 → **`env.user.step(질문)`으로 tau-bench 사용자 시뮬레이터에 직접 질의** →
  사용자가 승인하면 실행, 거부하면 수정하도록 에이전트에 반환.
- **Mechanism 2 (targeted reflection):** 동일 호출에서 정책 reminder 1줄을 관측에 주입.
- 비-mutating 행동은 게이트를 완전히 우회(논문과 동일).
- 기본 aux_model = main model(동일 모델 pairing, 논문 기본값). config에서 `aux_model:`로 override 가능.

### 구현상의 운영적 선택(충실성 메모)
논문은 사용자 피드백 후 **main model이 행동을 재발행**한다. 본 구현은 **확인 시 env가 동기 실행**한다:
- (a) 소형 main model(nano)이 tool-result로부터 정확한 tool call을 재발행하는 데 불안정,
- (b) tau-bench 사용자 시뮬레이터가 확인 턴에 `###STOP###`를 붙여 **에피소드를 조기 종료**시킴
  (초기 버그: 확인 직후 reward 0로 종료됨 — `###STOP###` 제거 + 동기 실행으로 해결).
**결과는 동일**(승인→실행, 거부→수정). 확인 턴은 에피소드를 종료시키지 않도록 `###STOP###`를 strip.
이미 한 번 게이팅된 동일 (tool,args)는 직접 실행(논문의 "post-feedback action executes directly" = anti-stall).

확인/거부 판정: 휴리스틱(yes/no 키워드) 우선, 모호하면 auxiliary에 yes/no 1콜.

## 4. 실행 방법

```bash
cd pipeline_code
./scripts/run_experiment.sh --method saber --end-index 30        # 충실 SABER
./scripts/run_experiment.sh --method saber_old --end-index 30    # 원래 단순 버전(참고)
```
출력: `data/results/saber_seed42.jsonl`. metadata에 `num_verifications`,
`num_executed_after_confirm`, `num_user_rejected` 기록. step_logs decision 값:
`saber_verify`(거부/확인 분기 전), `approve_after_user_confirm`, `reject_after_user`,
`execute_post_confirm`(재발행 직접 실행).

## 5. 결과 (30 task, seed 42) — P3_RESULTS.md §2 반영 완료
- **SABER(충실) pass@1 = 0.367** (n=30, 파싱 flake 0). user verification 47 / user reject 35 / 확인후 실행 31.
- **SABER(충실) 0.367 > Old SABER 0.300 > ours 0.233** — **사용자 게이팅 > 자율 LLM 게이팅**(논문 핵심 지지).
- 단 vanilla(0.400)·reflexion(0.567) 미만: 약한 user-sim(nano)이 과잉 거부(35/47)해 천장을 낮춤.
  논문은 강한 `claude-sonnet-4` user-sim 사용 → 확인 품질↑. (더 강한 OpenAI 모델로 향후 검증 가능.)
- 코드: `src/agents/saber_paper_agent.py`(SaberPaperAgent, factory에서 `saber`로 매핑) +
  `critic.saber_verify`·`critic.saber_user_confirmed` + `prompts.SABER_AUX_SYSTEM`.
  원래 단순 버전(SABERGatingAgent)은 `saber_old`로 보존.
</content>
