"""Small standalone SVG report from measured CSV, using only Python's stdlib."""

import argparse
import csv
import html
import json
from pathlib import Path


COLORS = {"LAND": "#977333", "HYBRID": "#db7937", "WATER": "#1686b3"}


def write_report(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "physics.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    stride = max(1, len(rows) // 2500)
    selected = rows[::stride]
    if selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="840" viewBox="0 0 1100 840">',
           '<rect width="1100" height="840" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#263238}.title{font-size:20px}.axis{stroke:#c9d0d5;fill:none}</style>',
           '<text class="title" x="35" y="30">Native-physics river crossing</text>',
           '<text x="35" y="53">Result: ' + html.escape(summary["result"]) + '; crossing passed: ' + str(summary["crossing_passed"]) + '</text>']

    def panel(x0, y0, w, h, title, xx, yy, series):
        xlo, xhi = xx; ylo, yhi = yy
        if xhi - xlo < 1e-8:
            xhi = xlo + 1
        if yhi - ylo < 1e-8:
            yhi = ylo + 1
        def p(x, y):
            return x0 + (float(x) - xlo) / (xhi - xlo) * w, y0 + h - (float(y) - ylo) / (yhi - ylo) * h
        svg.append(f'<text x="{x0}" y="{y0 - 12}">{html.escape(title)}</text>')
        svg.append(f'<rect class="axis" x="{x0}" y="{y0}" width="{w}" height="{h}"/>')
        for frac in [0., .5, 1.]:
            x = xlo + frac * (xhi - xlo); y = ylo + frac * (yhi - ylo)
            svg.append(f'<text x="{x0 + frac*w - 12:.1f}" y="{y0+h+18}">{x:.2f}</text>')
            svg.append(f'<text x="{x0-43}" y="{y0+h-frac*h+4:.1f}">{y:.2f}</text>')
        for index, (label, color, values) in enumerate(series):
            points = ' '.join(f'{a:.2f},{b:.2f}' for a,b in [p(x,y) for x,y in values])
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{points}"/>')
            svg.append(f'<text x="{x0 + index*150}" y="{y0+h+38}" style="fill:{color}">{html.escape(label)}</text>')

    times = [float(r["time_s"]) for r in selected]
    xx = (times[0], times[-1])
    positions = [(float(r["x_m"]), float(r["y_m"])) for r in selected]
    ys = [p[1] for p in positions]; xs = [p[0] for p in positions]
    # XY plot uses equal scales so small lateral drift is not visually exaggerated.
    center_y = (max(ys) + min(ys)) / 2
    xspan = max(24., max(xs) - min(xs) + 2.)
    yspan = xspan * 220 / 430
    panel(65, 95, 430, 220, "Trajectory: world X/Y (m)", (min(xs)-1, min(xs)-1+xspan),
          (center_y-yspan/2, center_y+yspan/2), [("measured XY", "#1686b3", positions)])
    panel(625, 95, 430, 220, "Body origin height: time (s) / Z (m)", xx,
          (min(-.1, min(float(r["z_m"]) for r in selected)-.1), max(float(r["z_m"]) for r in selected)+.1),
          [("body origin", "#485a6a", [(r["time_s"], r["z_m"]) for r in selected]),
           ("water level", "#1686b3", [(xx[0], 0), (xx[1], 0)])])
    force_max = max(float(r["buoyancy_n"]) for r in selected)
    force_max = max(force_max, max(float(r["left_normal_n"]) + float(r["right_normal_n"]) for r in selected))
    panel(65, 410, 430, 220, "Support forces: time (s) / force (N)", xx, (0, max(500., force_max*1.08)),
          [("buoyancy", "#1686b3", [(r["time_s"], r["buoyancy_n"]) for r in selected]),
           ("track normal load", "#977333", [(r["time_s"], float(r["left_normal_n"])+float(r["right_normal_n"])) for r in selected])])
    speed = [(float(r["time_s"]), (float(r["vx_m_s"])**2+float(r["vy_m_s"])**2)**.5) for r in selected]
    panel(625, 410, 430, 220, "Horizontal speed: time (s) / speed (m/s)", xx, (0, max(.8, max(y for x,y in speed)*1.1)),
          [("measured speed", "#485a6a", speed)])
    svg.append('<text x="65" y="715">Actuator modes over time (measured physics telemetry)</text>')
    for a, b in zip(selected[:-1], selected[1:]):
        x = 65 + (float(a["time_s"])-xx[0]) / max(1e-9, xx[1]-xx[0]) * 990
        width = (float(b["time_s"])-float(a["time_s"])) / max(1e-9, xx[1]-xx[0]) * 990
        svg.append(f'<rect x="{x:.2f}" y="730" width="{width+.3:.2f}" height="18" fill="{COLORS.get(a["mode"], "#bbb")}"/>')
    svg.append('<text x="65" y="775">Brown: LAND. Orange: HYBRID. Blue: WATER. Positions and forces are simulator ground truth.</text>')
    svg.append('<text x="65" y="800">Prototype parameters are uncalibrated. This is a motion test, without a learned navigation policy.</text></svg>')
    target = directory / "crossing_report.svg"
    target.write_text('\n'.join(svg), encoding="utf-8")
    return str(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(write_report(parser.parse_args().directory))
