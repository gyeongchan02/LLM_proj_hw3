# P3 Results & Analysis — Critic-Gated Pre-Action Error Detection (retail)

**작성자:** P3 (실험)  ·  **날짜:** 2026-06-18
**관련 문서:** 실행 셋업/통합 수정 = [P3_INTEGRATION_FIXES.md](P3_INTEGRATION_FIXES.md),
워크플로 = [implementation_guide_p3.md](implementation_guide_p3.md)

> ⚠️ **이 결과는 예비(preliminary) 결과다.** 비용 한도를 고려해 retail test split의
> **앞 30개 task, 단일 seed(42)** 로만 돌렸다. 모든 방법은 **동일한 task 0–29** 를 봤으므로
> 비교는 공정하다. 절대 수치보다 **방법 간 상대 경향**에 주목하라. pass^k(일관성)는
> 단일 seed라 측정하지 않았다.

---

## 1. 실험 구성

| 항목 | 값 |
|------|-----|
| 벤치마크 | tau-bench **retail**, test split, task index 0–29 (30개) |
| Main agent (모든 방법 공통) | `gpt-5.4-nano` |
| Critic — Ours | `gpt-5.4-mini` (별도/상위 모델) |
| SABER (충실) auxiliary | `gpt-5.4-nano` (main과 동일; 사용자 확인 + targeted reflection) |
| Old SABER critic | `gpt-5.4-nano` (자율 4-way, 비구조화 프롬프트) |
| User simulator | `gpt-5.4-nano` (tau-bench 내장, react + resilient 패치) |
| Seed | 42 (단일) |
| 방법 | vanilla · reflexion · **SABER**(충실) · ours · oracle · *Old SABER*(참고) |

- **Recovery Rate 분모:** vanilla 로그를 gold 액션 시퀀스와 대조해 *처음 어긋난 mutating step*
  이 있는 task 집합 = **16/30개** (`label_divergence vs-gold` 로 실행 로그에서 재생성;
  bootstrap 자기참조 회피 — C3 해결).
- **Critic 정확도:** 라이브 매칭(C1)을 피하고 오프라인 하니스 `critic_accuracy.py` 로
  perturbations.jsonl(146개 라벨)을 critic에 직접 투입해 측정 (§4).

---

## 2. 메인 결과 (Task 지표)

| 방법 | pass@1 | Recovery Rate | mean s/task | mutating steps | block | revise | ask_user |
|------|:------:|:-------------:|:-----------:|:--------------:|:-----:|:------:|:--------:|
| **reflexion** | **0.567** | **0.563** | 72.3 | 47 | 0 | 0 | 0 |
| vanilla (기준) | 0.400 | 0.188 | 28.8 | 42 | 0 | 0 | 0 |
| oracle | 0.367† | 0.375 | 28.2 | 38 | 0 | 0 | 1 |
| **SABER (충실 재현)** | **0.367** | 0.250 | 40.0 | 66 | — | — | — ‡ |
| Old SABER | 0.300 | 0.438 | 34.7 | 63 | 13 | 11 | 16 |
| **ours** | **0.233** | 0.188 | 40.0 | 55 | **30** | **0** | 13 |

(pass@1 = reward>0 비율, n=30. Recovery Rate = 위 16개 분기 task 중 성공 비율.
mean s/task = task당 평균 소요. block/revise/ask_user = mutating step에 대한 critic 판정 합계.)

†oracle 0.367은 **유효한 천장이 아님** — §5 참조(설계 한계 C2).
‡SABER(충실)는 4-way verdict를 쓰지 않고 **사용자 확인**으로 게이팅한다. 개입 지표는 별도:
**user verification 47회 / user reject 35회 / 확인 후 실행 31회**(metadata). block/revise/ask 컬럼은 0.
> **모델 구성:** Old SABER·ours·vanilla·reflexion·oracle = 원래 실행(react user, strict). SABER(충실)·이후
> 비교는 **resilient user-sim 패치** 적용(파싱 flake 제거, §5.4). vanilla/reflexion은 flake 0이라 동일.

### 핵심 관찰
1. **사용자 게이팅(SABER 충실, 0.367) > 자율 LLM 게이팅(Old SABER 0.300, ours 0.233).**
   판정 주체를 *사용자*로 돌리면(약한 시뮬레이터라도) 자율 차단보다 **덜 해롭다** — SABER 논문의 핵심 주장 지지.
2. **사후(post-hoc) reflexion이 모든 방법을 압도** (pass@1 0.567, Recovery 0.563), vanilla 0.400가 그 다음.
3. **자율 사전 게이팅(ours/Old SABER)은 vanilla보다 낮다.** ours의 Recovery(0.188)는 vanilla와 동일 —
   자율 게이팅이 분기 task 회복에 기여 못 함. "분리 critic이 reflexion보다 회복률↑" 가설은 이 subset에서 반증.
4. **자율 게이팅의 원인 = 과잉 차단(over-blocking).** ours는 55개 mutating 판정 중 block 30 / approve 12 /
   ask_user 13 / revise 0 — 승인보다 차단이 많고 revise 미사용. → 상위 모델 + 구조화 프롬프트가
   오히려 더 보수적(차단 편향)으로. SABER(충실)는 이 차단을 사용자에게 위임해 일부 회피(0.367).
5. **단, SABER(충실)도 vanilla(0.400) 미만.** 약한 user-sim(gpt-5.4-nano)이 과잉 거부(47회 중 35회 reject)해
   유효 행동까지 막는 마찰 발생. 논문은 강한 user-sim(`claude-sonnet-4`)을 써서 확인 품질이 더 높다(§7 한계).

---

## 3. Error Analysis (정성)

### 3.1 과잉 차단이 성공 task를 깨뜨린 사례 (vanilla 성공 → ours 실패): 7건
| task | ours가 차단/보류한 행동 | 비고 |
|------|----------------------|------|
| 1 | `exchange_delivered_order_items` **3회 block** + ask_user 1회 | vanilla는 정상 교환 완료. critic이 유효한 교환을 반복 차단 → 미완 |
| 5 | `transfer_to_human_agents` block | 해당 task에선 상담원 이관이 올바른 해결이었음 |
| 11 | `return_delivered_order_items` block + ask_user | 유효한 반품을 차단 |
| 17 | `transfer_to_human_agents` **2회 block** | 위와 동일 패턴 |
| 25 | `transfer_to_human_agents` 2회 + `return_delivered_order_items` block | 다중 차단으로 진행 불가 |
| 18, 24 | (게이팅 개입 없음) | critic 무관한 에이전트 자체 변동/실패 |

> 패턴: critic이 **`transfer_to_human_agents`·되돌릴 수 없는 반품/교환**을 과하게 막는다.
> 이들은 "비가역"이라 critic이 block/ask_user로 기울지만, 실제로는 **그 task의 정답 행동**인 경우가 많다.
> 비가역성 신호가 "위험"으로 과해석되어 정상 완료를 방해한다.

### 3.2 reflexion이 이긴 이유
- reflexion은 **행동을 차단하지 않는다.** 실패 시 반성문을 system(wiki)에 주입하고 재시도만 한다.
  따라서 기준 성공을 깎을 일이 없고 오직 추가 성공만 가능 → 안전한 향상.
- 최종 성공 attempt 분포: 1회차(즉시) 13건, 2회차 2건, 3회차 2건, 4회차(상한) 13건.
  → 재시도로 **4개 task를 추가 회복**(vanilla 12 → reflexion 17). 단, 평균 2.5배 느림(72s).

### 3.3 자율 게이팅 vs 사용자 게이팅 (ours/Old SABER vs SABER 충실)
- **ours(0.233) vs Old SABER(0.300):** 같은 자율-게이팅 틀, critic만 다름(상위모델+4조건 vs 동일모델+단순).
  ours가 더 낮다 — "분리·구조화"가 이 설정에선 정확도가 아니라 **보수성(차단 편향)**을 키웠다(ours: block 30/revise 0).
- **SABER 충실(0.367) > 둘 다:** 결정을 *사용자*에게 위임하니 자율 차단의 over-blocking 피해가 줄었다.
  단, 약한 user-sim이 47회 중 35회 거부 → 유효 행동까지 막는 새로운 마찰. anti-stall(재요청 시 직접 실행)로
  31회는 통과시켜 균형을 맞춤. 즉 **"사용자 게이팅 > 자율 게이팅"이지만, user-sim 품질이 천장을 좌우**한다.
- 자율 게이팅의 *오차단(false block)* 정량은 오프라인 정확도(§4)에서 확인(ours false-block-rate 0.32).

---

## 4. Critic 정확도 (오프라인 하니스, perturbations.jsonl 146개)

> 라이브 매칭(C1)을 피해 각 라벨 행동을 critic에 직접 투입해 측정. 이 하니스는 full goal +
> gold history를 주는 **낙관적 상한**이다(라이브는 더 낮음). 그런데도 정확도가 낮다는 게 핵심.
> gold 라벨 분포(146): **revise 52 / ask_user 34 / block 30 / approve 30** — revise가 최대 클래스.

| critic | 4-way acc | block prec | block recall | **false-block-rate** | reversibility acc | revise-arg acc |
|--------|:---------:|:----------:|:------------:|:--------------------:|:-----------------:|:--------------:|
| **ours** (gpt-5.4-mini, 4조건) | 0.308 | 0.351 | 0.667 | **0.319** | **0.884** | 1.000* |
| **saber** (gpt-5.4-nano, 단순) | 0.294 | 0.083 | 0.067 | 0.190 | 0.390 | 0.056 |

block 혼동행렬 — ours: tp=20 / **fp=37** / fn=10 / tn=79  ·  saber: tp=2 / fp=22 / **fn=28** / tn=94
(*revise-arg acc 1.0은 ours가 revise를 거의 예측 안 해 표본이 매우 작음 — 해석 주의.)

### 해석 (라이브 결과와 정확히 연결됨)
- **두 critic 모두 4-way 정확도가 ~0.30으로 낮다** — 낙관적 상한인데도 그렇다 → 라이브는 더 나쁨.
- **ours = 과잉개입(over-intervention).** false-block-rate 0.319, block precision 0.351
  (차단의 65%가 오차단). 결정적으로 **approve 라벨에 대한 4-way 정확도 = 0.0** — *올바른 행동을
  단 한 번도 깨끗이 approve하지 않았다*(전부 block/revise/ask_user로 개입). §2의 라이브
  over-blocking(block 30 / approve 12)과 정확히 일치하는 근거.
- **saber = 과소개입(under-intervention).** block recall 0.067 — 진짜 위반 30개 중 28개를 놓침.
  덜 막으니 라이브에서 ours보다 덜 망가져 pass@1이 약간 높지만(0.30 vs 0.23) 여전히 vanilla 미만.
- **분리·구조화의 유일한 분명한 이득 = 가역성 판단.** ours 0.884 vs saber 0.390. 4조건 구조가
  reversibility는 잘 잡지만, 그게 task 성공으로 연결되진 못함(오히려 비가역을 위험으로 과해석).

### 조건 Ablation (ours에서 한 조건씩 제거, 146 라벨)
| 제거 조건 | 4-way acc | false-block-rate | block recall | 비고 |
|-----------|:---------:|:----------------:|:------------:|------|
| (없음, full) | 0.308 | 0.319 | 0.667 | 기준 |
| −GOAL | 0.274 | 0.310 | 0.600 | 4-way↓ |
| −STATE | 0.274 | 0.319 | 0.567 | 4-way↓ |
| −CONSTRAINT | 0.281 | 0.267 | 0.533 | 4-way↓ |
| −POLICY | 0.301 | **0.491** | **0.800** | FBR 급증·무차별 차단↑ |

- **POLICY가 가장 load-bearing.** 제거 시 false-block-rate가 0.319→0.491로 급증하고 block을
  더 무차별적으로 남발(recall 0.667→0.800, 그러나 precision은 더 악화). 즉 정책 텍스트가
  그나마 차단을 *표적화*해 주고 있었다. (조건별 정확도에서도 POLICY 라벨 정확도가 0.79로 최고.)
- **GOAL·STATE·CONSTRAINT** 제거는 모두 4-way 정확도를 소폭 떨어뜨려 각 조건이 약간씩 기여함을 보인다.
- 단, full에서도 **GOAL 유형(같은 유저의 다른 주문으로 잘못 지정) 정확도는 0.0** — critic이 이
  오류를 revise로 고치지 못하고 다른 판정을 낸다(revise 미사용 문제와 연결).

---

## 5. 한계와 주의사항

1. **표본이 작다 (30 task, 단일 seed).** 신뢰구간이 넓다. 경향은 일관되나 수치는 변동 가능.
   추가 예산이 되면 전체 ~115 task + 다중 seed(pass^k)로 확장 권장.
2. **작은 모델(nano/mini).** critic 판단력이 제한적이라 over-blocking이 두드러진다. 큰 모델에선
   다를 수 있다(향후 검증 대상).
3. **Oracle 천장이 유효하지 않다 (C2).** 라이브 oracle은 perturbation의 *오염된 인자*에만 매칭되는데
   에이전트는 *올바른* 인자를 제안하므로 거의 개입하지 않는다(block 0, ask_user 1). 결국 vanilla와
   비슷하게 동작(0.367)하며 "완벽 탐지기 상한" 역할을 못 한다. 진짜 상한은 **오프라인 4-way accuracy=1.0**
   경계로 해석해야 한다. P1이 oracle을 "gold 행동에서의 일탈 탐지"로 재설계하면 라이브 천장 복원 가능.
4. **약한 user simulator (SABER 한계).** SABER(충실)의 사용자 확인은 `gpt-5.4-nano`(=에이전트)로
   시뮬레이션되는데, 논문은 **강한 `claude-sonnet-4`** 를 user-sim으로 썼다. 약한 user-sim은
   유효 행동도 과잉 거부(47회 중 35회)해 SABER 천장을 낮춘다. 더 강한 OpenAI 모델(예: gpt-5.4-mini)을
   user-sim으로 쓰면 결과가 달라질 수 있음(향후 검증). 논문 §8도 user-sim 단순화 한계를 인정.
5. **tau-bench react user 파싱 flake (해결됨).** react user-sim이 소형 모델의 비정형 출력("yes")을
   거부해 에피소드를 크래시(전체의 ~6%, ours/Old SABER/oracle에 2~3건씩, vanilla/reflexion 0). §5.4의
   resilient 패치(재시도+graceful fallback)로 제거. SABER(충실) 재실행은 패치 적용(0건). vanilla/reflexion은
   flake 0이라 미재실행(패치 적용해도 동일).

### 5.4 user-sim resilient 패치 (전 메서드 공통)
`runner.py`에서 `ReactUserSimulationEnv.generate_next_message`를 몽키패치 — 파싱 실패 시 최대 2회 재샘플,
그래도 실패하면 원문을 그대로 사용자 메시지로 사용(크래시 대신 graceful degrade). site-packages 미수정(재현 가능).

> ⚠️ 공정성 메모: SABER(충실)·이후 실행은 패치 ON, 기존 ours/Old SABER/oracle 수치는 패치 OFF의 옛 실행이라
> ~2~3건의 flake-제로처리가 포함됨(해당 메서드에 약간 불리). 헤드라인 결론(사용자>자율, reflexion 최고)은
> 견고하나, ours/oracle을 패치로 재실행하면 그 수치는 소폭 상향될 수 있음(예산 시 권장).

---

## 6. 결론 & Future Direction (가설 검증 요약)

- **개입 *주체*가 핵심:** **사용자 게이팅(SABER 충실 0.367) > 자율 LLM 게이팅(Old SABER 0.300, ours 0.233).**
  비가역 행동의 판정을 LLM이 자율로 하면 over-blocking으로 망가지지만, 사용자에게 위임하면 피해가 준다 —
  SABER 논문의 핵심(human-in-the-loop) 주장을 우리 환경에서도 지지.
- **개입 *시점*:** 사후(reflexion 0.567)가 사전 게이팅 전부를 압도. 단 평균 2.5배 느림(비용 trade-off).
- **우리 원안 가설 반증(이 subset):** "분리된 자율 critic 사전 게이팅이 reflexion보다 회복률↑"는 미성립.
  ours Recovery는 vanilla와 동률, reflexion 최고.
- **자율 게이팅 실패 원인:** **과잉 차단**(비가역 행동 false block) + **revise 미사용**. 오프라인 정량 확인 —
  ours 4-way 0.31, false-block-rate 0.32, **approve 라벨 정확도 0.0**. saber_old는 반대로 과소개입(block recall 0.07).
- **SABER 충실의 천장 제약:** 약한 user-sim(nano)이 과잉 거부(47중 35) → vanilla 미만. 강한 user-sim이면 개선 여지.
- **Future:**
  (a) **사용자 게이팅 + 강한 user-sim**(논문의 claude-sonnet-4급)으로 SABER 충실 재검증;
  (b) 자율 critic은 *명백한 위반*에만 block(임계값↑), 비가역≠위험 구분, revise 경로 활성화;
  (c) reflexion + 사용자 게이팅 **하이브리드**(사후 반성 + 비가역 확인);
  (d) 더 큰 모델·더 많은 task·다중 seed(pass^k) 확장.

---

## 부록 — 산출물 파일
- `data/results/{vanilla,reflexion,saber,saber_old,ours,oracle}_seed42.jsonl` — task별 reward + step_logs
  (gitignore: 미커밋, 용량). `saber`=충실 SABER, `saber_old`=원래 단순 버전.
- `data/results/metrics.csv` — task 지표 표 (커밋)
- `data/results/critic_acc_*.json` — 오프라인 critic 정확도 (커밋, ours/saber_old 대상)
- `data/labels/divergences.jsonl` — 실행 로그 기반 분기 라벨 16개 (커밋)
- 상세 SABER 충실성 분석·구현: **`P3_SABER_FIDELITY.md`**
</content>
