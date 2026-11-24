"""테스트 공통 설정: 합성 카메라용 ROI를 고정 (실제 로봇 설정과 분리).

환경 변수로 지정하므로 테스트가 따로 실행하는 스크립트(subprocess)에도 같은 설정이 적용된다.
"""

import os

os.environ["FLOOR_DIRT_PARAMS_OVERRIDE"] = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "params_test_override.yaml")
