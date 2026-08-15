# Aheui JIT 통계 기준선

검사 대상 프로그램마다 `.jitstats` 파일을 하나씩 둡니다. 형식은
`pyre/bench/synth/*.jitstats`와 같습니다. `majit_metainterp::JitStats`가
내놓는 카운터를 정렬된 `key=value` 행으로 기록합니다.

```
bridges_compiled=0
guard_failures=12
internal_compile_panics=0
loops_aborted=0
loops_compiled=3
```

## 실행 방법

```sh
cargo build -p aheui --release
python3 scripts/jitstats.py record             # 모든 기준선 갱신
python3 scripts/jitstats.py check              # check.sh의 5번 검사
python3 scripts/jitstats.py record logo/logo   # corpus 프로그램 하나만 갱신
python3 scripts/jitstats.py record pi/pi.jinseo
python3 scripts/jitstats.py record --jitstress pi/pi.jinseo
python3 scripts/jitstats.py sweep              # 여러 threshold에서 정확성 검사
python3 scripts/jitstats.py survey             # rpaheui corpus 조사, 판정 없음
python3 scripts/jitstats.py trend              # 프로그램별 실행 기록 비교
python3 scripts/opcensus.py record             # backend opcode 기준선 갱신
python3 scripts/opcensus.py check              # backend opcode 증가 검사
python3 scripts/opcensus.py show logo/logo     # 한 프로그램의 opcode 구성 출력
```

`check`는 통과 여부뿐 아니라 모든 대상 프로그램의 카운터를 출력합니다.
PASS만 출력해서는 JIT가 실제로 무엇을 했는지 알 수 없기 때문입니다. 기준선과
달라진 카운터에는 허용 한계까지 남은 여유도 함께 표시하므로, 다음 변화에서
실패할 만큼 한계에 가까운 값도 미리 볼 수 있습니다.

중단된 프로그램에는 `Counters.ABORT_*` 세부 항목도 표시합니다. 이 값들은
진단용이며 기준선에 넣지 않습니다. `loops_aborted`와 같은 사건을 두 번
검사하지 않기 위해서입니다. `JitStats`에는 중단 총계만 있고 profiler의
`print_stats`는 `MAJIT_LOG`를 켜야 보이는데, 중단이 나타날 만큼 큰 작업에서
항상 켜기에는 너무 느립니다.

## corpus 조사와 jitstress

`survey`는 참조 출력이 있는 rpaheui corpus 프로그램을 모두 실행하고 카운터를
출력하되 판정하지 않습니다. `$AHEUI_SNIPPETS`나 이 저장소 옆의
`rpaheui/snippets` checkout을 볼 때 알맞은 방식입니다. 이런 소스는 그 컴퓨터가
마지막으로 받은 내용일 뿐이므로 다른 곳에서 같은 기준선을 재현할 수 없습니다.
반면 이 저장소에 고정된 `snippets/` submodule을 쓰면 `check`와 `record`가
`bench/corpus/<dir>/<stem>.jitstats`를 기준선으로 사용합니다.

실사용 back-edge threshold 1039에서는 고정된 62개 프로그램 가운데 **3개**가
threshold에 도달합니다(submodule `4961b05`, majit `2674bdcb06b`, aheui
`13e4eb9`).

| 프로그램 | loop | bridge | 중단 | guard | JIT / 미컴파일 | 출력 |
|---|---:|---:|---:|---:|---:|---:|
| `logo` | 1 | 1 | 0 | 201 | 0.46초 / 4.56초 | 996310, `ok` |
| `pi/pi.jinseo` | 3 | 1 | 0 | 674 | 0.04초 / 0.63초 | 1005, `+nl` |
| `standard/loop` | 1 | 0 | 0 | 1 | 0.006초 / 0.14초 | 1, `ok` |

나머지는 실사용 threshold에 도달하지 않아 아무것도 컴파일하지 않습니다. 이들의
실사용 기준선은 모두 0이며, 문제가 생겼음을 나타내는 필드만 검사합니다.

`pi/pi.jinseo`는 merge point 동작을 검사하는 프로그램이며, 어떤 corpus를
쓰느냐에 따라 측정 내용도 달라집니다. 고정된 `aheui/snippets` 판은 1006바이트를
출력합니다. 같은 알고리즘의 한 fork는 자릿수가 더 커서 15001바이트를 출력하고
미컴파일 상태에서 123초가 걸립니다. 이 fork는 고정된 입력이 아니므로 검사하지
않지만, 정식 `pi/pi.jinseo` 행이 중요한 이유를 잘 보여 줍니다.

고정 corpus에는 `MAJIT_THRESHOLD=50`으로 실행하는 두 번째 검사 축도 있습니다.
기준선은 `bench/jitstress/<dir>/<stem>.jitstats`에 둡니다. 이는
`pyre/check.py`의 `*_jitstress` 행과 같은 방식으로, 동일한 프로그램을 더 낮은
threshold에서 실행하고 JIT와 미컴파일 실행의 출력을 바이트 단위로 비교합니다.

고정 corpus에서 실사용 값 `1039`는 3개 프로그램에서 JIT를 가동했고,
`pi/pi.jinseo`가 `compile_trace`의 기존 procedure token으로 향하는 JUMP를
밟은 횟수는 0이었습니다. `200`에서는 7개와 5회, `50`에서는 10개와 8회였습니다.
`10`은 이 JUMP 경로가 7회로 오히려 줄었습니다. `50` 검사에서는 중단과 출력
불일치가 모두 0이었으므로, 고정되지 않은 큰 fork를 검사 대상으로 삼지 않고도
실사용 threshold의 빈틈을 메웁니다.

corpus와 jitstress 검사는 참조 출력이 있는 고정 snippet을 전부 사용합니다.
대체 경로에서 찾은 snippet은 어떤 commit인지 보장할 수 없으므로 survey에만
사용합니다.

## 기준선 없는 threshold sweep

threshold가 바뀌면 같은 프로그램에서도 서로 다른 trace 모양이 만들어집니다.
하나의 기준선만 검사하면 특정 threshold에서만 드러나는 오컴파일을 놓칠 수
있습니다. `sweep`은 여러 `MAJIT_THRESHOLD` 값에서 JIT 실행과 threshold를
도달 불가능하게 높인 실행을 비교합니다. stdout과 종료 코드는 완전히 같아야
하며, JIT 통계 기준선은 사용하지 않습니다.

이 축은 실제로 threshold 28과 40에서는 맞지만 32에서는 틀렸던 `huntcook` 같은
결함을 찾기 위한 것입니다. `check.sh`는 고정 corpus가 있을 때 이 sweep도
실행합니다.

## backend opcode census

`scripts/opcensus.py`는 `MAJIT_LOG=1`, `MAJIT_THRESHOLD=50`으로 trace를 만들고
backend가 내보낸 기계 수준 연산의 수를 기록합니다. threshold 50은 jitstress
축과 같은 설정이므로 기준선도 `bench/jitstress/<디렉토리>/<이름>.opcensus`로,
같은 실행을 다른 계측기로 읽은 `.jitstats` 옆에 둡니다. `bench/` 아래 디렉토리는
기준선을 만든 **설정**을, 확장자는 그것을 읽은 **계측기**를 가리킵니다.

- `op.*`와 `total_ops`는 줄어들 수 있지만 늘어나면 실패합니다. 더 짧은 trace가
  목표이기 때문입니다.
- `out_bytes`와 `exit`는 정확히 같아야 합니다.
- `traces`는 진단용으로 표시하되 판정하지 않습니다.

새 기준선은 `scripts/opcensus.py record`로 만들고, `check.sh`에서는
`scripts/opcensus.py check`가 증가 여부를 검사합니다.

## 실행 기록과 `trend`

기준선은 JIT가 *지금* 무엇을 하는지는 말해 주지만 직전 실행과 무엇이
달라졌는지는 알려 주지 않습니다. 매번 예전 tree를 다시 빌드하지 않고 변화의
원인을 찾기 위해 실행 기록을 남깁니다. `check`와 `survey`는 실행한 프로그램마다
한 행씩 자동으로 추가합니다.

```sh
python3 scripts/jitstats.py check -m "merge-point segmenting trigger"
python3 scripts/jitstats.py trend                 # 모든 프로그램, 변화한 지점만
python3 scripts/jitstats.py trend pi/pi.jinseo    # 프로그램 하나
python3 scripts/jitstats.py trend logo/logo --all # 변화가 없는 행도 전부
python3 scripts/jitstats.py check --no-log        # 이번 실행은 기록하지 않음
```

`check`는 변경 뒤에 실행할 기본 명령입니다. 고정 submodule이 있으면 corpus와
jitstress 양쪽을 검사하고, 대체 소스를 사용하면 판정 없는 survey를 기록합니다.

survey의 corpus 출력은 각 프로그램 옆의 `.out`과 비교해 `ok`, `+nl`, `≠` 세
상태로 표시합니다. `+nl`은 `.out`에만 마지막 줄바꿈이 있는 경우입니다. 몇몇
프로그램은 표준 입력을 읽지만 survey가 입력을 주지 않으므로 `bahmanghui`처럼
읽지 못한 정수에 대해 `-1`을 출력할 수도 있습니다. `+nl`을 `ok`로 합치면 실제
줄바꿈 변화도 숨기게 되므로 별도 상태로 둡니다. 고정 corpus와 jitstress는 더
엄격합니다. 기준선에 stdout SHA prefix와 종료 코드를 저장하고 어느 쪽이든
바뀌면 실패합니다. 이때 JIT와 미컴파일 실행 비교도 완전 일치여야 합니다.

각 기록에는 카운터, 전체 `ABORT_*` 항목, `mc_diag` 감소 census, stdout 길이와
SHA 및 종료 코드, 실행 시간, `-m` 메모와 함께 **aheui와 majit 양쪽 commit**이
들어갑니다. `aheui-jit`는 `../../majit/*`를 path dependency로 사용하므로 이
수치들은 대개 majit 변경의 영향을 받습니다. 어느 한쪽 commit만 적으면 나중에
원인을 구분할 수 없습니다. `-dirty` 접미사는 commit하지 않은 변경이 있었다는
뜻입니다. `MAJIT_*`와 `AHEUI_*` 환경 변수도 함께 기록하므로 서로 다른 knob로
실행한 행을 잘못 비교하지 않습니다.

행에는 실행 당시 tree의 commit이 적히지만 binary는 그보다 전에 빌드되었을 수
있습니다. 그사이에 majit tree가 움직이면 그 binary를 만들지 않은 commit에
수치를 귀속하는 셈입니다. 이를 막기 위해 binary 수정 시각을 양쪽 HEAD commit
시각과 비교하고 `stale_build`를 기록합니다. `check`는 이를 크게 알리고
`trend`도 해당 행을 표시합니다. 그런 행을 해석하지 말고 다시 빌드해 실행해야
합니다.

`trend`는 값이 바뀐 실행만 출력하고 `Δ`로 차이를 표시합니다. 공유 컴퓨터에서는
시간 변동이 너무 크므로 실행 시간은 변화 판정에서 제외합니다. `ABORT_*`는
기준선으로 검사하지 않지만 변화 판정에는 포함합니다. 예를 들어
`abort_too_long 784 -> 7`은 수정의 모양을 잘 보여 주지만, 이를 합친
`loops_aborted` 총계는 다른 이유로도 움직일 수 있습니다.

기록 파일은 gitignore된 `bench/history.jsonl`입니다. 실행 시간은 컴퓨터마다
다르고 여러 worktree가 매번 같은 파일에서 충돌해서는 안 되기 때문입니다.
`AHEUI_JITSTATS_HISTORY`로 공유 경로를 지정할 수 있습니다. 오래 보존할 가치가
있는 결과는 위 survey 표에 직접 반영합니다. ledger는 자주 쌓이는 원자료이고,
표는 추려 낸 기록입니다.

## 각 카운터의 판정 방향

`pyre/check.py`의 `_jit_stats_regression_floor`와 같은 규칙을 사용합니다. 실사용
corpus와 jitstress는 같은 방향으로 판정하며, jitstress만 JIT threshold와
기준선 파일이 다릅니다.

| 카운터 | 실패 조건 |
|---|---|
| `loops_aborted` | 기준선보다 증가 |
| `internal_compile_panics` | 증가(정상 값은 0) |
| `guard_failures` | `base + max(base // 4, 2)` 초과 |
| `loops_compiled` | 기준선보다 감소 |
| `bridges_compiled` | 판정하지 않음. 일반적인 조정에서도 양방향으로 움직임 |

기준선은 목표가 아니라 현재 JIT 동작의 기록입니다. 이미 알려진 감소를 고정해
새로운 감소를 드러냅니다. 수치가 움직인 이유를 이해했고 의도한 변화일 때만
다시 기록해야 합니다.

## 정확성 검사

카운터를 비교하기 전에 고정 corpus의 모든 프로그램을 두 번 실행합니다. 한 번은
평소대로 실행하고, 다른 한 번은 `MAJIT_THRESHOLD`를 도달할 수 없을 만큼 높여
tracer가 작동하지 않게 합니다. 두 실행의 stdout과 종료 코드가 완전히 같아야
합니다. 이 검사가 없으면 오컴파일한 실행도 기준선 수치만으로 통과할 수 있습니다.

여기서는 `--no-jit` 대신 큰 `MAJIT_THRESHOLD`를 사용해야 합니다. `--no-jit`는
다른 interpreter인 `aheuinterpreter`를 선택하고 `naive` build에서만 쓸 수
있습니다. 큰 threshold는 tracer만 쉬게 하고 양쪽 모두 같은
`aheui_jit::mainloop` 경로를 유지합니다.
