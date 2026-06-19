# analysis_archive — 구(舊) 분석 보관

여기 있는 파일은 **약한 user simulator(`gpt-5.4-nano`)** 기준의 이전 분석이다. 약한 user-sim이
모든 점수를 인위적으로 낮추는 교란변수임이 드러나, 헤드라인 분석은 **강한 user-sim(`gpt-5.4`)** 기준의
3-method 비교(루트 [`../P3_RESULTS.md`](../P3_RESULTS.md))로 대체되었다.

- `P3_RESULTS_nano_6method.md` — nano user-sim, 6-method(vanilla/reflexion/saber/saber_old/ours/oracle)
  전체 분석. Recovery, Oracle 한계(C2), Old SABER, reflexion, 조건 ablation 등 상세 포함.
- `metrics_nano_6method.csv` — 위 분석의 task 지표.
- `critic_accuracy_summary.csv` — ours/Old SABER critic의 오프라인 정확도(4-way / false-block-rate /
  조건 ablation). **user-sim과 무관(offline)** 하므로 ours 과잉차단 설명에 현재도 유효.

> 참고로 사용할 것 — 현재 결론은 루트 `P3_RESULTS.md` 기준.
</content>
