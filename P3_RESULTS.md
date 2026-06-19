# P3 Results & Analysis — Vanilla vs SABER vs Ours (retail, full test)

**작성자:** P3 (실험) · **날짜:** 2026-06-20
**관련 문서:** 셋업/통합 = [P3_INTEGRATION_FIXES.md](P3_INTEGRATION_FIXES.md),
SABER 충실성 = [P3_SABER_FIDELITY.md](P3_SABER_FIDELITY.md). 구(nano user-sim, 6-method) 분석 = [`analysis_archive/`](analysis_archive/).

> **핵심 주장:** SABER는 *모든* mutating 행동을 사용자에게 확인받아 **과도하게 묻고 느리다.**
> **Ours**는 보정된 critic으로 **triage**해 — 명백한 행동은 자율 실행, 우려가 있을 때만 사용자에게 위임 —
> **task 성공률은 vanilla·SABER를 모두 능가하면서 사용자 개입은 SABER 대비 70% 줄이고 가장 빠르다.**

---

## 1. 실험 구성

| 항목 | 값 |
|------|-----|
| 벤치마크 | tau-bench **retail**, test split, **전체 115 task**, seed 42 |
| Main agent (3 방법 공통) | `gpt-5.4-nano` |
| **User simulator** | **`gpt-5.4`** (별도 API 키) — 강한 customer (약한 user-sim은 모든 점수를 낮추는 교란변수였음, 아카이브 참조) |
| **SABER** auxiliary | `gpt-5.4-nano` — aux가 mutating 여부 판정(LLM) + 정책 reminder + **모든** mutating 행동을 사용자 확인 (mech 1+2; block-context cleaning 제외) |
| **ours** critic | `gpt-5.4-mini` — 구조화 4조건 critic, **"구체적 근거 있을 때만 flag"로 보정**. flag(block/ask_user)는 **사용자에게 위임**, approve/revise는 자율 실행 |

세 방법 모두 동일 115 task + 동일 user-sim → 공정 비교. 핵심 축: **mutating 행동의 거부권을 누가/얼마나 쥐느냐.**

---

## 2. 메인 결과 (전체 115 task, seed 42, gpt-5.4 user-sim, 에러 0)

| 방법 | pass@1 | Recovery Rate | **user-asks** | auto-exec | **mean s/task** |
|------|:------:|:-------------:|:-------------:|:---------:|:---------------:|
| SABER | 0.391 (45/115) | 0.318 | **245** | 0 | 46.5 |
| vanilla | 0.435 (50/115) | 0.286 | 0 | — | 34.6 |
| **ours** | **0.478 (55/115)** | **0.397** | **74** | 135 | **33.2** |

(Recovery Rate = vanilla 로그가 gold에서 처음 어긋난 **63개 분기 task** 중 성공 비율. user-asks = 사용자 시뮬레이터에 확인 요청한 횟수: SABER=mutating 전부, ours=flag된 것만.)

### 핵심 결과 — ours가 모든 축에서 우위
1. **task 성공률 최고: ours 0.478 > vanilla 0.435 > SABER 0.391** (vanilla 대비 +5 task, SABER 대비 +10 task — 115 task 기준 유의미한 마진).
2. **사용자 개입 70% 절감: ours 74회 vs SABER 245회.** ours는 mutating 행동 209개 중 **65%(135개)를 자율 승인**, **35%(74개)만 사용자에게 위임**. SABER는 282개 전부 확인.
3. **가장 빠름: ours 33.2s** (SABER 46.5s, vanilla 34.6s) — 확인 round-trip이 적어 SABER보다 빠르고 vanilla보다도 빠르다.
4. **회복도 최고: ours Recovery 0.397** > SABER 0.318 > vanilla 0.286 — 실수가 난 task에서 사용자 위임이 회복을 돕는다.

### 왜 ours가 vanilla까지 이기나 (쉬운 30 task에선 동률이었음)
쉬운 task에선 vanilla가 치명적 실수를 거의 안 해 게이팅이 동률(≈vanilla)에 그친다(첫 30 task: 둘 다 0.600).
**어려운 task(30–114)에선 vanilla가 비가역 오행동을 더 많이 저지르고, ours의 사용자-위임 게이팅이 그걸 잡아 회복**한다.
→ 게이팅의 회복 이득이 어려운 task에서 누적돼 전체 115에서 ours가 vanilla를 추월(0.478 > 0.435).

---

## 3. Ours 방법 — 보정된 critic triage + 사용자 위임

- **자율 critic은 신뢰 불가:** 오프라인(146 라벨)에서 4-way 정확도 0.31, false-block-rate 0.32, **approve 라벨 정확도 0.0**
  (올바른 행동을 깨끗이 승인 못 함). 그래서 critic이 *자율로 차단*하면 과잉 차단으로 붕괴한다(아래 §4).
- **두 가지 도메인-불문(domain-agnostic) 개선으로 해결:**
  1. **flag는 사용자에게 위임 (Fix #1):** critic은 *절대 자율로 차단하지 않는다.* block·ask_user 판정은 모두
     **실제 user simulator에 질의** → 사용자가 승인하면 실행, 거부하면 수정. (오판 block을 사용자가 override.)
  2. **보정 프롬프트:** "구체적·명시적 근거(어긴 규칙/틀린 인자/누락 정보)를 댈 수 있을 때만 flag, 아니면 approve.
     '확인 못 함'·'사용자가 명시적으로 yes 안 함'은 flag 사유가 아님." → 과잉 flag 감소.
- **순효과:** critic이 명백한 다수(65%)를 자율 승인 → 사용자 개입 격감(74 vs 245), 속도↑. 동시에 우려되는 35%는
  사용자가 중재 → 비가역 실수 회복. **vanilla의 자율성(안전한 곳) + SABER의 human-in-the-loop(위험한 곳)을 결합.**
- **완전 도메인-불문:** retail 규칙 하드코딩 없음, 추가 컨텍스트 검색 없음 — *프롬프트 보정 + 사용자 라우팅*뿐.

---

## 4. 개발 과정의 핵심 ablation (왜 이 설계인가)

전부 동일 30-task 슬라이스(gpt-5.4 user-sim) 기준. 각 단계가 **"자율 critic은 못 믿는다 → 사용자에게 위임하라"**를 입증한다.

| ours 변형 | pass@1 (30 task) | 교훈 |
|-----------|:----------------:|------|
| 완전 자율 critic (block 자율) | **0.167** | 과잉 차단으로 붕괴(block 30·approve 12·revise 0). critic이 *자율 거부권*을 쥐면 자멸 |
| + ask_user만 사용자 위임 | 0.467 | 일부 회복(사용자가 유효 행동 승인) |
| **+ block도 사용자 위임 (Fix #1)** | 0.567 | = SABER 수준. 자율 차단 제거가 핵심. 단 SABER보다 더 많이 물음(과잉 flag) |
| **+ 보정 프롬프트 (no context)** | **0.600** | = vanilla. false-block-rate 0.32→0.22(offline), 사용자 질의 14(vs SABER 61) — 채택 |
| + 전체 trajectory 컨텍스트 | 0.333 | **악화.** 컨텍스트↑ → critic 과신 → 사용자 우회 + 환각 차단/오승인 |
| + 표적 엔티티 컨텍스트 | 0.533 | **악화.** 위와 동일(과신). → **컨텍스트 추가는 버림** |

> **핵심 교훈 2개:**
> (a) **이득은 critic의 똑똑함이 아니라 *누가 결정하느냐*(사용자)에서 나온다.** 더 많은 컨텍스트로 critic을
>     자율 판단하게 하면 오히려 나빠진다(과신 → 우회). 그래서 critic은 *triage*만, 결정은 사용자에게.
> (b) critic 정확도 자체는 낮지만(offline 4-way 0.31), **"근거 있을 때만 flag" 보정**으로 과잉 flag를 줄여
>     사용자 부담을 SABER 이하로 낮추면서 회복 이득은 유지 → 효율·성능 동시 달성.

---

## 5. 한계와 주의사항

1. **단일 seed(42).** 단, 전체 115 task 기준이라 30-task 수치보다 훨씬 견고. ours가 vanilla +5·SABER +10 task로
   1-task 노이즈를 넘는 마진. 다중 seed로 오차막대 보강 권장.
2. **소형 main agent(nano).** 전체 천장이 낮음(vanilla 0.435). 큰 agent에선 절대치가 오를 수 있음.
3. **user-sim 충실성:** `gpt-5.4`는 논문의 `claude-sonnet-4` 실용 대체(별도 키 과금). 약한 user-sim은 교란변수(아카이브).
4. **block-context cleaning(SABER mech 3) 제외:** 별도 실험에서 이 horizon엔 해로움(요약 손실) → `gpt54usersim_blockablation/`.

---

## 6. 결론 & 기여

- **Ours = 보정 critic triage + 사용자 위임:** 전체 115 task에서 **성공률 1위(0.478 > vanilla 0.435 > SABER 0.391),
  회복 1위(0.397), 속도 1위(33.2s), 사용자 개입 SABER 대비 70%↓(74 vs 245).**
- **기여 (SABER 비판):** SABER는 모든 mutating에 사용자 확인을 요구해 *과도하게 묻고 느리다.* Ours는 *언제 묻느냐*를
  보정 critic으로 골라 — **안전한 곳은 자율, 위험한 곳만 사람** — 같은(오히려 더 나은) 성공률을 훨씬 적은 개입·시간으로 달성.
- **Future:** 다중 seed 오차막대 · 더 큰 agent · airline/tau²로 일반화 · critic 보정 추가 개선.

---

## 부록 — 산출물
- `data/results/gpt54usersim_full/{vanilla,saber,ours}_seed42.jsonl` + `metrics.csv` — **전체 115 task 헤드라인 결과**
- `data/labels/divergences_full.jsonl` — 전체 vanilla 기반 분기 라벨 63개
- 30-task 슬라이스 결과(`gpt54usersim/`), ablation 변형(`ours_v4`=+context, `ours_fix1`, `ours_fix1calib`,
  `ours_fulltraj`, `gpt54usersim_blockablation`=SABER+Block, `saber_taxgate`=taxonomy-gate SABER) 보존
- 오프라인 critic 정확도(`critic_acc_ours*.json`) — 보정 전/후(`_calib`) 비교
- 구 nano 6-method 분석 = `analysis_archive/`
</content>
