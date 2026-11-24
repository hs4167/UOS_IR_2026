# UOS_IR_2026: 청소 구역 모니터링 로봇

서울시립대학교 인공지능학과 지능형로봇 기말 과제 프로젝트 (7조)

> 로봇이 강의실을 돌며 **청소 직후 바닥과 달라진 곳을 오염 구역으로 찾고**, 청소 후 다시 찾아가 **치워졌는지 검증**합니다.

## 파이프라인

```
[0] 준비 (1회)       카메라 보정 + 바닥 투영 행렬(호모그래피)
[1] 지도 작성 (1회)  LiDAR SLAM → 강의실 지도
[2] 기준 주행        청소 직후 주행 → 칸별 바닥 색 평균·변동폭 저장 (기준 바닥 지도)
[3] 정찰 주행        사용 후 주행 → 기준과 비교 → 사람·의자 등 물체 제외
                     → 칸별 오염 확률 누적 → 오염 구역 지도 (위치·크기·사진)
[사람]               오염 구역 청소
[4] 재방문·검증      오염 구역을 최단 순서로 방문 → 기준 대비 복원율로 처리 여부 판정
```

## 하드웨어 · 소프트웨어

| 항목 | 사양 |
| ---- | ---- |
| 로봇 | TurtleBot3 Waffle Pi (OpenCR, 360° 2D LiDAR LDS-02) |
| 컴퓨터 | ASUS PE1000N (NVIDIA Jetson Xavier NX 8GB) |
| 카메라 | Logitech C920 USB 웹캠 (바닥 촬영용, 자동 노출·초점 끄고 고정) |
| OS / JetPack | Ubuntu 20.04 (L4T r35.6) / JetPack 5.1.5 |
| ROS | ROS 2 Foxy |
| Python | 3.8 (conda 환경 `test1001`) |
| 주요 라이브러리 | OpenCV 4.10 (CUDA), PyTorch 2.1, jetson-inference (segNet, detectNet) |

## 디렉터리 구조

GitHub 첫 화면에 보이는 순서대로 정리했습니다. *(예정)* 표시는 아직 비어 있거나 앞으로 추가될 항목입니다.

```
UOS_IR_2026/
├── config/                 설정 파일
│   └── params.yaml         모든 처리 파라미터 (ROI, 격자 해상도, 비교 임계값, 확률, 재방문, 디버그 출력)
│   ├── camera_settings.yaml  웹캠 노출·초점·화이트밸런스 고정값 (기준·정찰 공통)
│   ├── amcl.yaml           위치 추정(AMCL)·지도 서버 설정
│                           (보정 후 생성) camera_intrinsics.yaml, floor_homography.yaml
├── docs/                   문서
│   └── 실험_기록.md        시도·결과·실패 원인 기록 (보고서 자료)
├── floor_dirt/             처리 로직 패키지 (순수 Python, ROS 의존 없음)
│   ├── config.py           params.yaml 로드, 부분 덮어쓰기, "compare.k" 형태 조회
│   ├── timing.py           처리 단계별 소요 시간(ms)을 프레임마다 CSV로 기록
│   ├── dataset.py          프레임 데이터셋 저장·읽기 (이미지 + 촬영 시점 자세 + LiDAR 점)
│   ├── projection.py       바닥 투영: 왜곡 보정, 픽셀 → 로봇 → 지도 좌표, 이미지를 바닥 격자로 옮기기
│   ├── calibration.py      카메라 보정: 체커보드 검출, 내부 보정(K, D), 바닥 투영 행렬(H), 오차 검사
│   ├── floor_grid.py       기준 바닥 격자: 칸별 색(Lab) 평균·분산·관측 횟수 누적, 저장·불러오기, 빈 곳 확인
│   ├── compare.py          기준과 비교: 칸별 차이(밝기/밝기+색/색), 조명 보정, 1칸 어긋남 흡수, 칸별 임계값
│   ├── masks.py            제외 마스크: LiDAR 점 주변, 로봇 몸체, detectNet 박스, segNet 바닥 아닌 픽셀
│   ├── dirt_map.py         오염 확률 지도: 칸별 log-odds 누적, 깨끗함/의심/확인 불가, 오염 구역 추출·저장
│   ├── viz.py              시각화: 결과 지도 dirt_map.png (LiDAR 지도 위 확률·구역·로봇), 구역 사진, 카메라 디버그 오버레이
│   ├── debug_stream.py     디버그 영상 출력: 없음 / 파일 / 브라우저(HTTP) / RTSP(젯슨)
│   ├── pipeline.py         처리 묶음: 기준 주행(ReferenceBuilder)·정찰 주행(Patroller). 오프라인·ROS 공용
│   ├── offline.py          오프라인 실행: 데이터셋 → 기준 지도 / 정찰 결과, 정답 비교 평가
│   ├── web.py              결과 웹 서버 (주행 선택, 오염 지도, 구역 표·사진, 평가, 처리 시간)
│   ├── synthetic.py        합성 카메라·강의실 장면 (테스트와 WSL 확인용)
│   ├── monitor.py          실시간 모니터 화면 합성: 카메라(+ROI·LiDAR 바닥 위치), LiDAR 위에서 본 그림, 상태 줄
│   ├── mjpeg_server.py     브라우저용 실시간 영상 서버 (MJPEG over HTTP, 표준 라이브러리만)
│   ├── jetson.py           jetson-inference/utils 불러오기 (conda에서 시스템 경로 보완)
│   ├── camera_settings.py  웹캠 설정: v4l2-ctl 컨트롤 읽기, 자동 끄기→값 순서로 적용 계획
│   ├── recording.py        녹화 로직: 저장 주기, 사진·스캔 시각 짝짓기, TF 대기열
│   ├── occupancy.py        LiDAR 지도: OccupancyGrid → pgm/yaml 저장, 지도 뷰어 그림
│   ├── tegralog.py         젯슨 자원 기록: 녹화·처리 중 tegrastats.log 저장, 메모리·CPU·GPU·온도 요약
│   └── revisit.py          재방문: 최단 방문 순서, 설 자리(벽 피하기), 구역별 복원율 판정, 기준 갱신(백업)
├── launch/                 ROS 2 launch 파일
│   ├── robot_lds02.launch.py  bringup (LDS-02 LiDAR용, 설치된 bringup이 LDS-01 드라이버만 띄워서 따로 만듦)
│   └── localization.launch.py 위치 추정: 저장된 지도 + AMCL (RViz 없음)
├── maps/                   지도
│   └── reference/          기준 바닥 지도 저장 위치 (reference.npz + reference.yaml, 기준 주행 후 생성)
│                           (SLAM 후 생성) classroom.pgm / classroom.yaml
├── ros_nodes/              ROS 2 노드 (토픽 구독·발행만 하고 처리는 floor_dirt 호출)
│   ├── camera_node.py      웹캠 → /image_raw (시작 시 고정 설정 적용)
│   ├── dataset_recorder.py 주행 녹화: 사진 + 찍은 순간의 위치(TF) + LiDAR → data/<run>/
│   ├── map_viewer.py       지도 뷰어 (RViz 대신): 지도 + 로봇 + LiDAR → 브라우저 :8082
│   └── map_saver.py        /map → maps/<이름>.pgm/yaml (기존 지도 백업)
├── scripts/                실행 스크립트
│   ├── robot.sh            환경 설정 + TurtleBot3 bringup (LDS-02 기본, source 하면 환경 설정만)
│   ├── teleop.sh           키보드 주행
│   ├── odom_monitor.sh     오도메트리 확인 (수신 주기, 위치, 속도, 이동 거리)
│   ├── monitor.sh          실시간 모니터 실행 (conda 유지 + ROS 환경 설정 → live_monitor.py)
│   ├── ros_conda_env.sh    우리 ROS 노드용 환경 설정 (monitor·camera·record가 공유)
│   ├── camera.sh           카메라 노드 실행
│   ├── record.sh           주행 녹화 실행 (data/<run 이름>/)
│   ├── camera_settings.py  웹캠 노출·초점 고정 적용·조정·저장
│   ├── slam.sh             SLAM 지도 작성 (slam_toolbox)
│   ├── save_map.sh         지도 저장 → maps/
│   ├── map_view.sh         지도 뷰어 실행
│   ├── localize.sh         위치 추정 (지도 + AMCL, 시작 위치 지정)
│   ├── set_pose.sh         위치 추정에 현재 위치 다시 알려주기
│   ├── live_monitor.py     카메라 + LiDAR + odom을 한 화면으로 송출 (노트북 브라우저로 확인)
│   ├── capture_images.py   웹캠으로 보정용 사진 촬영 (화면 없이 자동/Enter 촬영)
│   ├── calibrate_intrinsics.py  체커보드 사진들 → camera_intrinsics.yaml
│   ├── calibrate_floor.py  바닥 보드 사진 + 실측 좌표 → floor_homography.yaml
│   └── check_floor_projection.py  다른 위치 실측점으로 투영 오차 검사 (통과/실패)
│                           (예정) 재방문
├── tests/                  floor_dirt 단위 테스트
│   ├── test_foundation.py  config·timing·dataset 테스트
│   ├── test_projection.py  투영 테스트 (좌표 규칙, 왕복 오차, 격자 위치 1칸 이내, 유효 영역)
│   ├── test_calibration.py 보정 테스트 (K·D 복원, 보드 방향 정렬, H 정확도, 잘못 잰 원점 검출)
│   ├── test_floor_grid.py  기준 격자 테스트 (평균·분산 정확도, 주행 병합, 저장, 지도 방향, 합성 주행)
│   ├── test_compare.py     비교 테스트 (얼룩·색종이 검출, 어긋남·조명 변화 오탐 없음, 합성 정찰)
│   ├── test_masks.py       제외 마스크 테스트 (박스·다각형·LiDAR 원, 가짜 jetson-inference, 물체 오탐 제거)
│   ├── test_dirt_map.py    오염 확률 테스트 (누적·클램프, 상태, 구역 크기·중심, 저장, 합성 정찰 구역 1개)
│   ├── test_viz.py         시각화 테스트 (색·방향·빗금·구역 표시, LiDAR 배경, 오버레이, 구역 사진, 디버그 출력)
│   ├── test_e2e.py         합성 강의실 E2E: 기준 주행 → 정찰 → 오염 3개 검출·오탐 0·위치 오차 2 cm 이내
│   ├── test_offline.py     오프라인 실행 테스트 (평가 짝짓기, 결과 웹, 스크립트 전체 흐름)
│   ├── test_recording.py   녹화 준비 테스트 (C920 컨트롤 해석·적용 순서, 주기·짝짓기·대기열, 노드 문법)
│   ├── test_tegralog.py    자원 기록 테스트 (젯슨 실제 출력 해석, 요약, 가짜 tegrastats, 웹 표시)
│   ├── test_revisit.py     재방문 테스트 (순서 최적성, 설 자리, 복원율, 기준 갱신, 합성 재방문 전체)
│   ├── test_mapping.py     지도 테스트 (저장한 SLAM 지도 방향, 뷰어 그림, AMCL 설정)
│   └── test_monitor.py     모니터 테스트 (LiDAR 그림 방향, 카메라 위 LiDAR 점, 상태 줄, 메시지 변환, 재생 실행, 브라우저 스트림)
├── .gitignore              git 제외 목록 (.venv/, data/, results/, 캐시 등)
├── README.md               이 문서
├── pytest.ini              테스트 설정 (tests/ 폴더 지정, floor_dirt import 경로)
└── requirements-dev.txt    개발 PC(WSL) 전용 패키지 목록. 젯슨에서는 사용 금지
```

녹화(`data/<run>/`)와 오프라인 처리(`results/<run>/`) 폴더에는 젯슨 자원 기록 `tegrastats.log`가 자동으로 생깁니다 (WSL에서는 생기지 않음).

`.gitkeep`은 빈 폴더를 git에 남기기 위한 빈 파일입니다. `data/`(주행 녹화·이미지)와 `results/`(실행 결과)는 용량이 커서 git에 올리지 않으며, 실행하면 로컬에 생깁니다.

## 사용법

### 로봇 주행 (젯슨)

**LDS-02 드라이버 설치 (처음 한 번, conda 끈 상태):**

```bash
mkdir -p ~/ld08_ws/src && cd ~/ld08_ws/src
git clone -b ros2-devel https://github.com/ROBOTIS-GIT/ld08_driver.git
cd ~/ld08_ws
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
```

**전원 켜는 순서:**

```
1. 로봇 배터리 스위치 ON
2. OpenCR RESET 버튼 → "도레미파솔라시도" 시작음 확인 (로봇을 움직이지 않음)
3. bringup 실행
```

conda를 끈 상태에서 저장소 폴더로 이동한 뒤 실행합니다.

| 터미널 | 명령 | 역할 |
| ------ | ---- | ---- |
| 1 | `./scripts/robot.sh` | 환경 설정 + bringup (OpenCR, LiDAR) |
| 2 | `./scripts/teleop.sh` | 키보드 주행 |
| 3 | `./scripts/odom_monitor.sh` | 오도메트리 확인 |
| 4 | `conda activate test1001` → `./scripts/monitor.sh` | 실시간 모니터 |

### 실시간 모니터링 (젯슨 → 노트북 브라우저)

**젯슨 터미널 1 (bringup):**

```bash
./scripts/robot.sh
```

**젯슨 터미널 2 (모니터):**

```bash
conda activate test1001
./scripts/monitor.sh
```

**노트북 (Edge/Chrome 주소창):**

```
http://<젯슨IP>:8080
```

**옵션:**

```bash
./scripts/monitor.sh --no-ros                     # 카메라만 (bringup 없이)
./scripts/monitor.sh --topic /image_raw           # 카메라 노드의 이미지 토픽 사용
./scripts/monitor.sh --no-camera                  # LiDAR·odom만
./scripts/monitor.sh --camera /dev/video2         # 다른 카메라 장치
./scripts/monitor.sh --fps 5 --quality 60         # 네트워크가 느릴 때
./scripts/monitor.sh --output rtsp                # VLC로 보기: rtsp://<젯슨IP>:8554/monitor
./scripts/monitor.sh --help
```

**개발 PC (WSL, 데이터셋 재생 → Windows 브라우저 http://localhost:8080):**

```bash
.venv/bin/python scripts/live_monitor.py --dataset data/<run> --loop
```

### 지도 작성 · 위치 추정 (젯슨)

**지도 작성 (터미널 1 bringup, 터미널 2 키보드 주행은 위와 같음):**

```bash
./scripts/slam.sh                          # 터미널 3 (conda 끔)
conda activate test1001 && ./scripts/map_view.sh   # 터미널 4 → 브라우저 http://<젯슨IP>:8082
conda activate test1001 && ./scripts/save_map.sh   # 다 그렸으면 (maps/classroom.*)
```

**위치 추정 (로봇을 SLAM 시작 자리·방향에 두고):**

```bash
./scripts/localize.sh                      # conda 끔
./scripts/localize.sh --x 1.2 --y -0.5 --yaw 90
./scripts/set_pose.sh 0 0 0                # 위치가 어긋났을 때
```

### 주행 녹화 (젯슨)

**카메라 설정 고정 (처음 한 번, 값은 모니터 보며 조정):**

```bash
conda activate test1001
python scripts/camera_settings.py --exposure 250 --focus 30
python scripts/camera_settings.py --exposure 250 --focus 30 --save
```

**터미널 1 (bringup, conda 끔):**

```bash
./scripts/robot.sh
```

**터미널 2 (카메라 노드):**

```bash
conda activate test1001
./scripts/camera.sh
```

**터미널 3 (모니터, 선택):**

```bash
conda activate test1001
./scripts/monitor.sh --topic /image_raw
```

**터미널 4 (녹화, 끝나면 Ctrl+C):**

```bash
conda activate test1001
./scripts/record.sh test_odom --frame odom                     # 지도 없이 시험
./scripts/record.sh 2026-11-25_ref --note "형광등 전부"          # 지도 + 위치 추정 켠 상태
```

**터미널 5 (키보드 주행, conda 끔):**

```bash
./scripts/teleop.sh
```

### 카메라 보정 (젯슨, conda `test1001`)

체커보드 칸 크기를 자로 재서 `config/params.yaml`의 `calibration.square`에 넣은 뒤 진행합니다.

```bash
# 1) 내부 보정: 보드를 화면 구석구석·여러 각도로 움직이며 20장
python scripts/capture_images.py --out data/calib_intrinsics --count 20 --pattern 9x6
python scripts/calibrate_intrinsics.py --images "data/calib_intrinsics/*.png"
# 2) 바닥 보정: 보드를 로봇 앞 바닥에 놓고 기준 코너(가장 가깝고 오른쪽) 좌표를 줄자로 잼
python scripts/capture_images.py --out data/calib_floor --manual --count 2
python scripts/calibrate_floor.py --image data/calib_floor/000.png --origin 0.250 -0.090
# 3) 검사: 보드를 다른 위치로 옮겨 찍은 사진으로 오차 평균 1 cm 이하인지 확인
python scripts/check_floor_projection.py --board data/calib_floor/001.png --origin 0.380 0.020
```

### 오프라인 처리 · 결과 웹

**기준 주행 → 기준 바닥 지도:**

```bash
python scripts/run_reference.py --run data/<기준 run> --map maps/classroom.yaml
```

**정찰 주행 → 오염 구역:**

```bash
python scripts/run_patrol.py --run data/<정찰 run> --map maps/classroom.yaml
```

**정답과 비교:**

```bash
python scripts/evaluate.py --results results/<정찰 run> --truth data/<정찰 run>/truth.json
```

**재방문 계획 → (청소 후 재방문 녹화) → 판정:**

```bash
python scripts/plan_revisit.py --results results/<정찰 run> --map maps/classroom.yaml
python scripts/run_revisit.py --run data/<재방문 run> --results results/<정찰 run>
python scripts/run_revisit.py --run data/<재방문 run> --results results/<정찰 run> --promote   # 처리 완료 구역을 새 기준으로
```

**결과 웹 (브라우저 http://<주소>:8000):**

```bash
python scripts/serve_results.py
```

**개발 PC (WSL, 합성 데이터로 전체 흐름):**

```bash
.venv/bin/python scripts/make_synthetic_run.py
.venv/bin/python scripts/run_reference.py --run data/synthetic_ref --map data/synthetic_map/classroom.yaml --calib-dir data/synthetic_calib --no-dnn --out data/synthetic_reference
.venv/bin/python scripts/serve_results.py &
.venv/bin/python scripts/run_patrol.py --run data/synthetic_patrol --map data/synthetic_map/classroom.yaml --reference data/synthetic_reference --calib-dir data/synthetic_calib --no-dnn --synthetic-detector
.venv/bin/python scripts/evaluate.py --results results/synthetic_patrol --truth data/synthetic_patrol/truth.json
.venv/bin/python scripts/plan_revisit.py --results results/synthetic_patrol --map data/synthetic_map/classroom.yaml --start 0.35 1.32 0
.venv/bin/python scripts/run_revisit.py --run data/synthetic_revisit --results results/synthetic_patrol --reference data/synthetic_reference --calib-dir data/synthetic_calib --no-dnn
```

### 개발환경 테스트

**젯슨:**

```bash
conda activate test1001
python -m pytest -q
```

**개발 PC (WSL):**

```bash
uv venv -p 3.8 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

## 진행 상황

| 단계 | 내용 | 상태 |
| ---- | ---- | ---- |
| 기반 | 파라미터, 처리 시간 측정, 프레임 데이터셋 | ✅ |
| 주행 | bringup, 키보드 주행, 오도메트리 확인 (젯슨) | ✅ |
| [0] | 카메라 보정, 바닥 투영 | 🟨 코드 완료 (WSL 합성 검증), 실기 보정 전 |
| [1] | LiDAR 연결, SLAM 지도 작성 | 🟨 LiDAR(LDS-02)·모니터·SLAM/위치 추정 스크립트 완료, 실제 지도 작성 남음 |
| [2] | 기준 바닥 지도 | 🟨 처리·실행 스크립트 완료 (WSL 합성 검증), 실제 주행 녹화 남음 |
| [3] | 오염 확률 지도, 오염 구역 추출 | 🟨 처리·실행·평가·결과 웹 완료 (WSL 합성: 3/3 검출, 오탐 0), 실제 주행 남음 |
| [4] | 재방문, 처리 여부 검증 | 🟨 계획·판정·기준 갱신 완료 (WSL 합성: 3구역 모두 정답), 재방문 노드(Nav2) 남음 |

## References

- Bormann et al., 2013 — 청소 로봇의 오염 탐지 (문제 설정, 평가 방법)
- Lee et al., 2016, *Sensors* — 단안 카메라 바닥 투영
- Pathmakumar et al., 2021 — 청소 검증
- Thrun et al., *Probabilistic Robotics*, 9장 — 점유 격자 지도
- [ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3), [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)
