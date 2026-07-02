"""HTML rendering for recruiter candidate cards."""

from __future__ import annotations

from .candidate_card import CandidateCard

CARD_CSS = """
<style>
.shortlist-wrap { font-family: Inter, system-ui, sans-serif; }
.candidate-card {
  background: #1e1f24;
  border: 1px solid #2d2f36;
  border-radius: 14px;
  padding: 18px 20px;
  margin-bottom: 16px;
}
.card-header { display: flex; align-items: flex-start; gap: 14px; }
.rank-badge {
  background: #e86a2f;
  color: white;
  font-weight: 700;
  width: 34px; height: 34px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  font-size: 15px;
}
.card-title { flex: 1; }
.card-name { font-size: 20px; font-weight: 700; color: #f2f3f5; margin: 0 0 4px 0; }
.card-subtitle { color: #a8adb8; font-size: 14px; margin: 0; }
.card-score { text-align: right; min-width: 72px; }
.score-num { color: #3ecf8e; font-size: 22px; font-weight: 700; }
.score-bar { height: 4px; background: #2d2f36; border-radius: 4px; margin-top: 6px; overflow: hidden; }
.score-fill { height: 100%; background: linear-gradient(90deg, #2ecc71, #3ecf8e); border-radius: 4px; }
.tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 10px 0; }
.tag {
  font-size: 12px; padding: 4px 10px; border-radius: 999px;
  border: 1px solid transparent; white-space: nowrap;
}
.tag-active { background: #1a3a5c; color: #6eb6ff; border-color: #2a5a8a; }
.tag-otw { background: #12351f; color: #5ddea0; border-color: #1f6b3f; }
.tag-notice { background: #12351f; color: #5ddea0; border-color: #1f6b3f; }
.tag-loc { background: #2a2c33; color: #c5c9d3; }
.tag-must { background: #12351f; color: #5ddea0; border-color: #1f6b3f; }
.tag-skill { background: #2a2c33; color: #d7dae2; }
.card-summary { color: #c8ccd6; font-size: 14px; line-height: 1.55; margin: 8px 0 0 0; }
</style>
"""


def render_card_html(card: CandidateCard) -> str:
    pct = int(card.score * 100)
    tags = [f'<span class="tag tag-active">{card.active_label}</span>']
    if card.open_to_work:
        tags.append('<span class="tag tag-otw">Open to work</span>')
    tags.append(f'<span class="tag tag-notice">Notice: {card.notice_days}d</span>')
    tags.append(f'<span class="tag tag-loc">📍 {card.location}</span>')
    if card.all_must_haves:
        tags.append('<span class="tag tag-must">All must-haves</span>')
    for skill in card.skill_tags:
        tags.append(f'<span class="tag tag-skill">{skill}</span>')

    # Render badges from badge_system
    badge_html = ""
    if hasattr(card, "badges") and card.badges:
        from .badge_system import render_badges_html
        badge_html = render_badges_html(card.badges)

    # Enhanced detail line
    detail_html = ""
    if hasattr(card, "enhanced_detail") and card.enhanced_detail:
        safe_detail = card.enhanced_detail.replace("₹", "<span style='font-family: sans-serif; margin-right: 2px;'>₹</span>")
        detail_html = f'<p style="color:#6b7280;font-size:12px;margin:6px 0 0 0;line-height:1.4">{safe_detail}</p>'

    safe_summary = card.summary.replace("₹", "<span style='font-family: sans-serif; margin-right: 2px;'>₹</span>")

    return f"""
<div class="candidate-card">
  <div class="card-header">
    <div class="rank-badge">{card.rank}</div>
    <div class="card-title">
      <p class="card-name">{card.name}</p>
      <p class="card-subtitle">{card.title} · {card.years:.1f}yr · {card.company}</p>
    </div>
    <div class="card-score">
      <div class="score-num">{pct}%</div>
      <div class="score-bar"><div class="score-fill" style="width:{pct}%"></div></div>
    </div>
  </div>
  {badge_html}
  <div class="tag-row">{''.join(tags)}</div>
  <p class="card-summary">{safe_summary}</p>
  {detail_html}
</div>
"""


def render_shortlist_html(cards: list[CandidateCard], title: str = "AI-powered candidate ranking system") -> str:
    body = "\n".join(render_card_html(c) for c in cards)
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>{CARD_CSS}</head><body class='shortlist-wrap'><h2 style='color:#f2f3f5'>{title}</h2>{body}</body></html>"
