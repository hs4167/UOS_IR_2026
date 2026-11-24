#!/usr/bin/env python3
"""보정용 체커보드 인쇄 파일 (A4, SVG) 만들기. 설치할 것 없음.

SVG는 크기를 mm로 적으므로 '실제 크기(100%)'로 인쇄하면 칸 크기가 정확하다.
인쇄 후 반드시 자로 재서 config/params.yaml의 calibration.square에 넣는다 (프린터마다 조금씩 다름).

사용 예
  python scripts/make_checkerboard.py                  # 안쪽 코너 9x6, 칸 20 mm → results/checkerboard_9x6_20mm.svg
  python scripts/make_checkerboard.py --square 25      # 칸 25 mm
인쇄: 브라우저로 SVG 열기 → 인쇄 → 용지 A4, 배율 '실제 크기' 또는 100% (페이지에 맞춤 X), 여백 없음/최소
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floor_dirt.config import REPO_ROOT, load_params  # noqa: E402

A4 = (210.0, 297.0)  # mm (가로, 세로)


def checkerboard_svg(cols, rows, square_mm, page=A4, landscape=True):
    """안쪽 코너 cols x rows (칸은 cols+1 x rows+1) 체커보드 SVG 문자열."""
    pw, ph = (page[1], page[0]) if landscape else page
    nx, ny = cols + 1, rows + 1
    bw, bh = nx * square_mm, ny * square_mm
    if bw + 2 * square_mm > pw or bh + 2 * square_mm > ph:
        raise ValueError("A4에 다 안 들어갑니다 (보드 {:.0f} x {:.0f} mm + 흰 여백). 칸을 줄이세요".format(bw, bh))
    x0, y0 = (pw - bw) / 2.0, (ph - bh) / 2.0 - 8.0       # 아래쪽에 자 확인선 자리
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{0}mm" height="{1}mm" viewBox="0 0 {0} {1}">'.format(pw, ph),
             '<rect width="{}" height="{}" fill="white"/>'.format(pw, ph)]
    for r in range(ny):
        for c in range(nx):
            if (r + c) % 2 == 0:
                parts.append('<rect x="{0:.3f}" y="{1:.3f}" width="{2}" height="{2}" fill="black"/>'.format(
                    x0 + c * square_mm, y0 + r * square_mm, square_mm))
    # 배율 확인용 100 mm 선 (인쇄 후 자로 재서 100 mm인지 확인)
    ly = y0 + bh + 10.0
    lx = (pw - 100.0) / 2.0
    parts.append('<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="black" stroke-width="0.4"/>'.format(lx, ly, lx + 100))
    for x in (lx, lx + 100):
        parts.append('<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="black" stroke-width="0.4"/>'.format(x, ly - 3, ly + 3))
    parts.append('<text x="{}" y="{}" font-size="3.5" text-anchor="middle" font-family="sans-serif">'
                 '이 선이 정확히 100 mm여야 함 | 안쪽 코너 {}x{} | 칸 {} mm (인쇄 후 자로 다시 잴 것)</text>'.format(
                     pw / 2.0, ly + 7, cols, rows, square_mm))
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    cal = load_params()["calibration"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="{}x{}".format(*cal["pattern"]), help="안쪽 코너 (가로x세로)")
    ap.add_argument("--square", type=float, default=20.0, help="칸 한 변 [mm]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    out = args.out or os.path.join(REPO_ROOT, "results", "checkerboard_{}x{}_{:g}mm.svg".format(cols, rows, args.square))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(checkerboard_svg(cols, rows, args.square))
    print("저장: {}".format(out))
    print("보드 크기 {:.0f} x {:.0f} mm (칸 {} x {}). 인쇄 후 칸 크기를 자로 재서 params.yaml calibration.square에 m 단위로"
          .format((cols + 1) * args.square, (rows + 1) * args.square, cols + 1, rows + 1))


if __name__ == "__main__":
    main()
