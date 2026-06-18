# P3 Integration Fixes — tau-bench compatibility

**작성자:** P3 (실험)  ·  **날짜:** 2026-06-18

실험을 돌리려고 `pipeline_code/`를 실제 설치된 tau-bench(`sierra-research/tau-bench`,
최신 main)에 연결했더니, 파이프라인 코드가 **가정했던 tau-bench API와 실제 API가 달라서
그대로는 한 task도 실행되지 않았다.** P1 코드의 로직(critic/프롬프트/taxonomy/metrics)은
그대로 두고, **tau-bench 연결부(glue)만** 최소 수정했다. 아래에 무엇을·왜 고쳤는지 정리한다.

> 요약: critic·프롬프트·데이터·지표 로직은 손대지 않았다. tau-bench의 `Env`/`Agent`
> 인터페이스에 맞추는 어댑터 계층만 수정했다. 모델 이름(`gpt-5.4-nano`/`gpt-5.4-mini`)은
> **실제 존재하는 모델이라 그대로 사용**(교수님 배포 키로 접근 가능).

---

## 실제 tau-bench API (설치본 기준, introspection으로 확인)

| 항목 | 실제 시그니처 / 반환 |
|------|------|
| `get_env(...)` | `get_env(env_name, user_strategy, user_model, task_split, user_provider=None, task_index=None)` |
| `Env.reset(task_index)` | → `EnvResetResponse(observation, info)` (pydantic) |
| `Env.step(action)` | → `EnvResponse(observation, reward, done, info)` (pydantic, **튜플 아님**) |
| `Action` | pydantic, 필드 `name`, `kwargs` |
| `ToolCallingAgent.__init__` | `(tools_info, wiki, model, provider, temperature=0.0)` |
| `Agent.solve(env, task_index, max_num_steps=30)` | → `SolveResult(reward, messages, info, total_cost)` |
| 에이전트 LLM 호출 | `litellm.completion(...)` (env 내부) |

핵심: 에이전트는 `env.tools_info` / `env.wiki`로 생성되고, `solve(env)` 안에서
`env.reset` / `env.step`를 호출하며 `EnvResponse`의 `.observation/.reward/.done/.info`를 읽는다.

---

## 고친 부분 (파일별)

### 1. `src/agents/gated_env.py` — 튜플 → `EnvResponse`
- `reset()`이 `EnvResetResponse`를 그대로 반환하고, `goal`은 `response.observation`에서 추출하도록 변경
  (이전엔 응답 객체 전체를 `str()` 처리). `solve()`가 `env.reset(task_index=...)`로 부르므로 키워드 인자 허용.
- `step()`의 모든 분기가 **`EnvResponse` 객체**를 반환하도록 변경:
  - approve/passthrough/revise → 실제 `env.step()`의 `EnvResponse`를 (revise는 observation에 피드백을 덧붙여) 반환.
  - block/ask_user → **합성 `EnvResponse`**(`observation`=차단 메시지, `reward=0.0`, `done=False`,
    `info`=직전 실제 `EnvInfo` 재사용 → `solve()`가 `info.model_dump()` 호출 가능).
- 에이전트가 읽는 `tools_info` / `wiki` 속성을 env로 forward.

### 2. `src/agents/vanilla.py` — 공용 빌더 + `solve()`
- `build_base_agent(env, model, provider, wiki=None)` 헬퍼 추가:
  `ToolCallingAgent(tools_info=env.tools_info, wiki=..., model=..., provider=...)`로 **올바르게** 생성.
  (tools_info/wiki는 env에서 와야 하므로 `__init__`이 아니라 `run()` 시점에 생성.)
- `run()`이 `agent.run(task=, env=)`(존재하지 않음) 대신 **`agent.solve(env=gated, task_index=...)`** 호출,
  `SolveResult.reward`로 보상 추출. `total_cost`도 메타데이터에 기록.

### 3. `critic_agent.py` / `saber_agent.py` / `oracle_agent.py`
- `__init__`에서 base agent를 미리 만들던 코드 제거(그땐 env가 없어서 불가능).
  `run()`에서 `build_base_agent(env, ...)` 후 `solve()` 호출하도록 통일. `total_cost` 기록.

### 4. `src/agents/reflexion_agent.py` — reflection 주입 위치 수정
- 기존: reflection을 **task.instruction** 앞에 붙임 → 그러나 `solve()`는 task를 env에서 읽고,
  task.instruction은 **유저 시뮬레이터**를 초기화하는 데 쓰임 → reflection이 엉뚱하게 "고객" 역할로 샘.
- 수정: reflection을 **에이전트의 wiki(시스템 프롬프트)** 앞에 붙여 `build_base_agent(..., wiki=...)`로 주입.
  이게 tau-bench에서 Reflexion을 올바르게 구현하는 방식(사후·동일모델·재시도 축은 그대로 유지).
- 실패 후 reflection 생성은 `SolveResult.messages` 대화 로그에서 수행.

### 5. `src/harness/runner.py`
- `get_env(... user_model_provider=...)` → 실제 키워드 **`user_provider=`** 로 수정 (그대로면 TypeError).
- `.env` 자동 로드(`load_dotenv()`) 추가 — `run_experiment.sh`(bash) 대신 Windows에서
  `python -m src.harness.runner`를 직접 호출하기 때문.
- `litellm.drop_params = True` 설정 — gpt-5.x 계열은 `temperature != 1` 등 일부 파라미터를 거부하는데,
  litellm이 미지원 파라미터를 조용히 드롭하게 함(에러 방지). litellm INFO 로그도 WARNING으로 낮춤.

---

## 안 고친 것 (의도적으로 그대로 둠)
- `src/critic/*` (프롬프트·critique 로직·파싱·fallback), `src/data/*`, `src/eval/*`의 **지표 계산 로직**.
- `metrics.py::compute_critic_metrics` 는 `integration_issues_p2.md` C1대로 **사용하지 않음**
  (critic 정확도는 `critic_accuracy.py` 오프라인 하니스로 측정).

## 실행 환경 메모
- Python 3.12 venv(`.venv/`, .gitignore에 포함). tau-bench 설치 시 Windows 한글 로케일 때문에
  `setup.py`가 `cp949`로 README를 읽다 실패 → `PYTHONUTF8=1` 환경변수로 해결.
- 모델: main/SABER-critic/user-sim = `gpt-5.4-nano`, Ours critic = `gpt-5.4-mini` (config 기본값 그대로).
- 비용(관측): vanilla ≈ \$0.003/task, ours ≈ \$0.0055/task (task당, 30-step 한도).
</content>
</invoke>
