# UOS_IR_2026 — 청소 구역 모니터링

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
| 로봇 | TurtleBot3 Waffle Pi (OpenCR, 360° 2D LiDAR) |
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
│                           (보정 후 생성) camera_intrinsics.yaml, floor_homography.yaml
│                           (예정) camera_settings.yaml
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
│   └── masks.py            제외 마스크: LiDAR 점 주변, 로봇 몸체, detectNet 박스, segNet 바닥 아닌 픽셀
│                           (예정) 오염 확률, 재방문
├── launch/                 (예정) ROS 2 launch 파일
├── maps/                   지도
│   └── reference/          기준 바닥 지도 저장 위치 (reference.npz + reference.yaml, 기준 주행 후 생성)
│                           (예정) classroom.pgm / classroom.yaml (LiDAR SLAM 지도)
├── ros_nodes/              (예정) ROS 2 노드 (토픽 구독·발행만 하고 처리는 floor_dirt 호출)
├── scripts/                실행 스크립트
│   ├── robot.sh            환경 설정 + TurtleBot3 bringup (source 하면 환경 설정만)
│   ├── teleop.sh           키보드 주행
│   ├── odom_monitor.sh     오도메트리 확인 (수신 주기, 위치, 속도, 이동 거리)
│   ├── capture_images.py   웹캠으로 보정용 사진 촬영 (화면 없이 자동/Enter 촬영)
│   ├── calibrate_intrinsics.py  체커보드 사진들 → camera_intrinsics.yaml
│   ├── calibrate_floor.py  바닥 보드 사진 + 실측 좌표 → floor_homography.yaml
│   └── check_floor_projection.py  다른 위치 실측점으로 투영 오차 검사 (통과/실패)
│                           (예정) 카메라 설정 고정, rosbag 추출, 오프라인 처리, 평가, 결과 웹 서버
├── tests/                  floor_dirt 단위 테스트
│   ├── synthetic.py        테스트용 합성 카메라 (가상 바닥·체커보드를 정해진 자세에서 찍은 이미지 생성)
│   ├── test_foundation.py  config·timing·dataset 테스트
│   ├── test_projection.py  투영 테스트 (좌표 규칙, 왕복 오차, 격자 위치 1칸 이내, 유효 영역)
│   ├── test_calibration.py 보정 테스트 (K·D 복원, 보드 방향 정렬, H 정확도, 잘못 잰 원점 검출)
│   ├── test_floor_grid.py  기준 격자 테스트 (평균·분산 정확도, 주행 병합, 저장, 지도 방향, 합성 주행)
│   ├── test_compare.py     비교 테스트 (얼룩·색종이 검출, 어긋남·조명 변화 오탐 없음, 합성 정찰)
│   └── test_masks.py       제외 마스크 테스트 (박스·다각형·LiDAR 원, 가짜 jetson-inference, 물체 오탐 제거)
├── .gitignore              git 제외 목록 (.venv/, data/, results/, 캐시 등)
├── README.md               이 문서
├── pytest.ini              테스트 설정 (tests/ 폴더 지정, floor_dirt import 경로)
└── requirements-dev.txt    개발 PC(WSL) 전용 패키지 목록. 젯슨에서는 사용 금지
```

`.gitkeep`은 빈 폴더를 git에 남기기 위한 빈 파일입니다. `data/`(주행 녹화·이미지)와 `results/`(실행 결과)는 용량이 커서 git에 올리지 않으며, 실행하면 로컬에 생깁니다.

## 사용법

### 로봇 주행 (젯슨)

conda를 끈 상태에서 저장소 폴더로 이동한 뒤 실행합니다.

| 터미널 | 명령 | 역할 |
| ------ | ---- | ---- |
| 1 | `./scripts/robot.sh` | 환경 설정 + bringup (OpenCR, LiDAR) |
| 2 | `./scripts/teleop.sh` | 키보드 주행 |
| 3 | `./scripts/odom_monitor.sh` | 오도메트리 확인 |

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
| [1] | LiDAR 연결, SLAM 지도 작성 | ⬜ |
| [2] | 기준 바닥 지도 | 🟨 격자 누적·저장 완료 (WSL), 실행 스크립트는 단위 11 |
| [3] | 오염 확률 지도, 오염 구역 추출 | ⬜ |
| [4] | 재방문, 처리 여부 검증 | ⬜ |

## References

- Bormann et al., 2013 — 청소 로봇의 오염 탐지 (문제 설정, 평가 방법)
- Lee et al., 2016, *Sensors* — 단안 카메라 바닥 투영
- Pathmakumar et al., 2021 — 청소 검증
- Thrun et al., *Probabilistic Robotics*, 9장 — 점유 격자 지도
- [ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3), [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)
