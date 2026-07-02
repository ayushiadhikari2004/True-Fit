"""Custom SVG visualizations — high-performance, responsive charts in dark mode.

Includes custom SVG Donut and SVG Pie chart renderers.
"""

from __future__ import annotations

import math


def draw_svg_donut(slices: list[tuple[str, float, str]]) -> str:
    """slices is a list of (label, percentage/count, hex_color)."""
    slices = [s for s in slices if s[1] > 0]
    if not slices:
        return "<p style='color:#6b7280;font-size:13px;text-align:center'>No data to display</p>"

    total = sum(s[1] for s in slices)
    if total == 0:
        return "<p style='color:#6b7280;font-size:13px;text-align:center'>No data to display</p>"

    cx, cy, r = 100, 100, 70
    stroke_width = 16

    current_angle = -90  # Start at top
    paths = []

    for label, val, color in slices:
        angle_delta = (val / total) * 360
        if angle_delta >= 359.9:
            paths.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke_width}" />'
            )
            continue

        next_angle = current_angle + angle_delta

        # Convert polar coordinates to Cartesian
        x1 = cx + r * math.cos(math.radians(current_angle))
        y1 = cy + r * math.sin(math.radians(current_angle))
        x2 = cx + r * math.cos(math.radians(next_angle))
        y2 = cy + r * math.sin(math.radians(next_angle))

        large_arc = 1 if angle_delta > 180 else 0

        path_d = f"M {x1} {y1} A {r} {r} 0 {large_arc} 1 {x2} {y2}"
        paths.append(
            f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="{stroke_width}" '
            f'stroke-linecap="round" class="donut-segment" />'
        )
        current_angle = next_angle

    svg_content = "\n".join(paths)
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#c5c9d3;margin:4px 0">'
        f'<div style="width:10px;height:10px;border-radius:2px;background:{color}"></div>'
        f'<div style="font-weight:500">{label}</div>'
        f'<div style="color:#6b7280;margin-left:auto">{val:.0f} ({val/total:.0%})</div>'
        f'</div>'
        for label, val, color in slices
    )

    return f"""
    <div style="display:flex;align-items:center;gap:24px;justify-content:center;padding:12px 0;flex-wrap:wrap">
      <svg width="140" height="140" viewBox="0 0 200 200">
        {svg_content}
      </svg>
      <div style="display:flex;flex-direction:column;justify-content:center;flex:1;min-width:140px">
        {legend_items}
      </div>
    </div>
    """


def draw_svg_pie(slices: list[tuple[str, float, str]]) -> str:
    """slices is a list of (label, percentage/count, hex_color)."""
    slices = [s for s in slices if s[1] > 0]
    if not slices:
        return "<p style='color:#6b7280;font-size:13px;text-align:center'>No data to display</p>"

    total = sum(s[1] for s in slices)
    if total == 0:
        return "<p style='color:#6b7280;font-size:13px;text-align:center'>No data to display</p>"

    cx, cy, r = 100, 100, 75
    current_angle = -90
    paths = []

    for label, val, color in slices:
        angle_delta = (val / total) * 360
        if angle_delta >= 359.9:
            paths.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" stroke="#1e1f26" stroke-width="2" />'
            )
            continue

        next_angle = current_angle + angle_delta

        # Arc endpoints
        x1 = cx + r * math.cos(math.radians(current_angle))
        y1 = cy + r * math.sin(math.radians(current_angle))
        x2 = cx + r * math.cos(math.radians(next_angle))
        y2 = cy + r * math.sin(math.radians(next_angle))

        large_arc = 1 if angle_delta > 180 else 0

        # Sector path
        path_d = f"M {cx} {cy} L {x1} {y1} A {r} {r} 0 {large_arc} 1 {x2} {y2} Z"
        paths.append(
            f'<path d="{path_d}" fill="{color}" stroke="#1e1f26" stroke-width="2" class="pie-slice" />'
        )
        current_angle = next_angle

    svg_content = "\n".join(paths)
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#c5c9d3;margin:4px 0">'
        f'<div style="width:10px;height:10px;border-radius:20%;background:{color}"></div>'
        f'<div style="font-weight:500">{label}</div>'
        f'<div style="color:#6b7280;margin-left:auto">{val:.0f} ({val/total:.0%})</div>'
        f'</div>'
        for label, val, color in slices
    )

    return f"""
    <div style="display:flex;align-items:center;gap:24px;justify-content:center;padding:12px 0;flex-wrap:wrap">
      <svg width="140" height="140" viewBox="0 0 200 200">
        {svg_content}
      </svg>
      <div style="display:flex;flex-direction:column;justify-content:center;flex:1;min-width:140px">
        {legend_items}
      </div>
    </div>
    """
