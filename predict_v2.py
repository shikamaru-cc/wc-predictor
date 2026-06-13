#!/usr/bin/env python3
"""
世界杯预测整合引擎 v2 — 从文件动态读取
- matches.json：所有已知比赛元数据
- poly_data.json：实时 Polymarket 赔率
- 运行 cup26matches 模型 → 融合 → 更新 HTML
"""
import subprocess, json, os, sys
from datetime import datetime

BASE_DIR = "/tmp/world-cup-2026-prediction-model"
REPO_DIR = "/root/wc-predictor"
ALPHA = 0.5  # 模型权重

# ============ DATA LOADING ============

def load_matches():
    """读取比赛元数据"""
    with open(os.path.join(REPO_DIR, "matches.json"), encoding="utf-8") as f:
        data = json.load(f)
    return data["matches"]

def load_poly_data():
    """读取 Polymarket 赔率数据"""
    path = os.path.join(REPO_DIR, "poly_data.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("matches", {})

def get_today_slugs():
    """获取当天比赛 slug 列表（基于 poly_data 或全部）"""
    poly = load_poly_data()
    return list(poly.keys())

# ============ MODEL ============

def run_model(team1, team2):
    """运行 cup26matches 模型"""
    result = subprocess.run(
        ["node", "predict.mjs", team1, team2],
        capture_output=True, text=True, cwd=BASE_DIR, timeout=30
    )
    lines = result.stdout.strip().split('\n')

    p_home = p_draw = p_away = None
    e_home = e_away = None
    elo_home = elo_away = None

    for line in lines:
        s = line.strip()
        # Parse "brazil (Elo 1955) vs morocco (Elo 1874)"
        m = __import__('re').search(r'Elo\s+(\d+)', s.split('vs')[0] if 'vs' in s else '')
        if m and elo_home is None:
            elo_home = int(m.group(1))
        m = __import__('re').search(r'Elo\s+(\d+)', s.split('vs')[-1] if 'vs' in s else '')
        if m and elo_away is None:
            elo_away = int(m.group(1))

        if 'win' in s and '%' in s:
            pct = float(s.split('%')[0].strip().split()[-1])
            if p_home is None:
                p_home = pct
            else:
                p_away = pct
        elif 'draw' in s and '%' in s:
            p_draw = float(s.split('%')[0].strip().split()[-1])
        elif 'expected goals' in s:
            parts = s.split(':')[1].strip().split('–')
            if len(parts) == 2:
                e_home = float(parts[0].strip())
                e_away = float(parts[1].strip())

    return (p_home, p_draw, p_away), (e_home, e_away), (elo_home, elo_away), result.stdout.strip()


def blend(model_probs, poly_probs, alpha=ALPHA):
    """加权融合模型和市场概率"""
    if poly_probs is None:
        return model_probs
    home = alpha * model_probs[0] + (1 - alpha) * (model_probs[0] + poly_probs[0]) / 2  # 标准化融合
    draw = alpha * model_probs[1] + (1 - alpha) * (model_probs[1] + poly_probs[1]) / 2
    away = alpha * model_probs[2] + (1 - alpha) * (model_probs[2] + poly_probs[2]) / 2
    total = home + draw + away
    return (round(home / total * 100, 1), round(draw / total * 100, 1), round(away / total * 100, 1))


# ============ HTML GENERATION ============

def generate_match_html(match, poly, model, expected, blended, elos):
    """生成单场详情页 HTML"""
    slug = match["slug"]
    name1, name2 = match["name1"], match["name2"]
    flag1, flag2 = match["flag1"], match["flag2"]
    group = match["group"]
    kickoff = match["time"]
    venue = match.get("venue", "")
    elo1, elo2 = elos or (0, 0)
    e_h, e_a = expected or (0, 0)
    b_h, b_d, b_a = blended

    # 概率条宽度
    total_w = b_h + b_d + b_a
    hw = round(b_h / total_w * 100, 1) if total_w else 33.3
    dw = round(b_d / total_w * 100, 1) if total_w else 33.3
    aw = round(b_a / total_w * 100, 1) if total_w else 33.3

    analysis = f"""
    <div class="ab"><p><strong>数据解读</strong></p>
    <p>{flag1} {name1}（Elo {elo1}）vs {flag2} {name2}（Elo {elo2}）</p>
    <p>模型 λ 预期进球：{e_h:.2f} — {e_a:.2f}</p>
    <p>融合胜率：{name1} {b_h}% / 平 {b_d}% / {name2} {b_a}%</p>
    </div>"""

    sources_html = ""
    if poly:
        p_h, p_d, p_a = poly
        sources_html = f"""
    <div class="sources">
      <div class="src"><div class="src-label">模型概率</div><div class="v">{model[0]}%</div><div class="l">平 {model[1]}% / {model[2]}%</div></div>
      <div class="src"><div class="src-label">市场概率</div><div class="v">{p_h}%</div><div class="l">平 {p_d}% / {p_a}%</div></div>
    </div>"""
    else:
        sources_html = f"""
    <div class="sources">
      <div class="src"><div class="src-label">模型概率</div><div class="v">{model[0]}%</div><div class="l">平 {model[1]}% / {model[2]}%</div></div>
    </div>"""

    delta_elo = elo1 - elo2
    delta_str = f"+{delta_elo}" if delta_elo > 0 else str(delta_elo)

    today_str = datetime.now().strftime("%m/%d")

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>{name1} vs {name2} — 世界杯数据解读</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f4ed;display:flex;justify-content:center;align-items:center;min-height:100vh;padding:12px}}
.card{{width:100%;max-width:375px;background:#faf9f5;border-radius:16px;padding:24px 18px 20px;box-shadow:rgba(0,0,0,0.05) 0px 4px 24px;border:1px solid #f0eee6}}
.hd{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px}}
.hd-left h1{{font-family:Georgia,serif;font-size:20px;font-weight:500;color:#141413;line-height:1.2}}
.hd-right{{text-align:right}}
.hd-right .date{{font-size:13px;font-weight:500;color:#c96442}}
.hd-right .info{{font-size:10px;color:#87867f}}
.disc{{background:#f5f4ed;border-radius:8px;padding:6px 10px;margin-bottom:16px;border:1px solid #e8e6dc}}
.disc p{{font-size:9px;color:#5e5d59;line-height:1.4}}
.top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}
.tag{{font-size:9px;font-weight:500;padding:2px 8px;border-radius:4px}}
.tag-g{{background:#e8e6dc;color:#5e5d59}}
.tag-f{{background:#c96442;color:#faf9f5}}
.top .t{{font-size:11px;color:#87867f}}
.top .v{{font-size:9px;color:#87867f;margin-top:1px;text-align:right}}
.teams{{display:flex;align-items:center;gap:8px;margin-bottom:10px}}
.tm{{flex:1;display:flex;align-items:center;gap:8px}}
.tm.r{{flex-direction:row-reverse;text-align:right}}
.fl{{font-size:36px;width:42px;text-align:center}}
.ti .n{{font-size:18px;font-weight:500;color:#141413}}
.ti .e{{font-size:10px;color:#87867f;margin-top:1px}}
.dlta{{text-align:center;min-width:56px}}
.dlta .a{{font-size:13px;font-weight:500;color:#c96442}}
.dlta .l{{font-size:8px;color:#87867f}}
.sources{{display:flex;gap:4px;margin-bottom:8px}}
.src{{flex:1;background:#f5f4ed;border-radius:6px;padding:4px 6px;text-align:center}}
.src .v{{font-size:10px;font-weight:500;color:#141413}}
.src .l{{font-size:7px;color:#87867f}}
.src .src-label{{font-size:7px;color:#87867f;margin-bottom:2px}}
.ob{{display:flex;align-items:center;justify-content:center;gap:6px;background:#f5f4ed;border-radius:8px;padding:6px 10px;margin-bottom:8px}}
.ob .ol{{font-size:8px;color:#87867f;text-transform:uppercase}}
.ob .ov{{font-size:13px;font-weight:500;color:#141413}}
.ob .od{{width:3px;height:3px;background:#d1cfc5;border-radius:50%}}
.pr{{display:flex;align-items:center;gap:4px;margin-bottom:8px}}
.pr .pl{{font-size:9px;color:#5e5d59;font-weight:500;min-width:36px;text-align:right}}
.pr .pl.r{{text-align:left}}
.prt{{flex:1;height:4px;background:#f0eee6;border-radius:4px;overflow:hidden;display:flex}}
.sh{{height:100%;background:#141413}}
.sd{{height:100%;background:#d1cfc5}}
.sa{{height:100%;background:#c96442}}
.ft{{margin-top:12px;padding-top:8px;border-top:1px solid #f0eee6;display:flex;justify-content:space-between;font-size:8px;color:#87867f}}
</style></head><body>
<div class="card">
<div class="hd">
<div class="hd-left"><h1>⚽ 世界杯数据</h1></div>
<div class="hd-right"><div class="date">{today_str}</div><div class="info">{group}</div></div>
</div>
<div class="disc"><p>纯数据分析，不构成投注建议</p></div>
<div class="top">
<div><span class="tag tag-g">{group}</span></div>
<div><div class="t">{kickoff}</div><div class="v">{venue}</div></div>
</div>
<div class="teams">
<div class="tm"><div class="fl">{flag1}</div><div class="ti"><div class="n">{name1}</div><div class="e">Elo {elo1} · 预期 {e_h:.2f}</div></div></div>
<div class="dlta"><div class="a">{delta_str}</div><div class="l">实力差</div></div>
<div class="tm r"><div class="fl">{flag2}</div><div class="ti"><div class="n">{name2}</div><div class="e">Elo {elo2} · 预期 {e_a:.2f}</div></div></div>
</div>
{sources_html}
<div class="ob">
<span class="ol">融合概率</span>
<span class="ov">{name1} {b_h}%</span><span class="od"></span>
<span class="ov">平 {b_d}%</span><span class="od"></span>
<span class="ov">{name2} {b_a}%</span>
</div>
<div class="pr">
<span class="pl">{b_h}%</span>
<div class="prt"><div class="sh" style="width:{hw}%"></div><div class="sd" style="width:{dw}%"></div><div class="sa" style="width:{aw}%"></div></div>
<span class="pl r">{b_a}%</span>
</div>
{analysis}
<div class="ft"><span>cup26matches + Polymarket</span><span>#世界杯 #数据解读</span></div>
</div>
</body></html>"""
    return html


def update_list_page(all_data):
    """更新 index.html 列表页"""
    idx_template = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>⚽ 世界杯数据解读</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f4ed;display:flex;justify-content:center;min-height:100vh;padding:12px}}
.page{{width:100%;max-width:375px}}
h1{{font-family:Georgia,serif;font-size:22px;font-weight:500;color:#141413;margin-bottom:4px;margin-top:4px}}
.date{{font-size:12px;color:#87867f;margin-bottom:12px}}
.disc{{background:#f5f4ed;border-radius:8px;padding:6px 10px;margin-bottom:12px;border:1px solid #e8e6dc}}
.disc p{{font-size:9px;color:#5e5d59;line-height:1.4}}
.item{{display:flex;align-items:center;gap:8px;background:#faf9f5;border-radius:12px;padding:10px 12px;margin-bottom:6px;border:1px solid #f0eee6;text-decoration:none;transition:all 0.1s}}
.item:active{{background:#f0eee6}}
.item .fl{{font-size:22px;width:28px;text-align:center}}
.item .info{{flex:1}}
.item .info .n{{font-size:14px;font-weight:500;color:#141413}}
.item .info .l{{font-size:9px;color:#87867f;margin-top:1px}}
.item .pct{{text-align:right}}
.item .pct .v{{font-size:11px;font-weight:500;color:#c96442}}
.item .pct .label{{font-size:7px;color:#87867f}}
.progress-bar{{height:2px;background:#f0eee6;border-radius:2px;overflow:hidden;display:flex;margin-top:4px}}
.progress-bar .seg{{height:100%}}
</style></head><body>
<div class="page">
<h1>⚽ 世界杯</h1>
<div class="date">{date}</div>
<div class="disc"><p>纯数据分析 · 不构成投注建议</p></div>
{items}
</div>
</body></html>"""

    items = []
    for d in all_data:
        m = d["match"]
        slug = m["slug"]
        b = d["blended"]
        flag1, flag2 = m["flag1"], m["flag2"]
        name1, name2 = m["name1"], m["name2"]
        group = m["group"]
        kickoff = m["time"]

        total = b[0] + b[1] + b[2]
        hw = round(b[0] / total * 100) if total else 33
        dw = round(b[1] / total * 100) if total else 33
        aw = round(b[2] / total * 100) if total else 34

        items.append(f"""<a class="item" href="match/{slug}.html">
  <div class="fl">{flag1}</div>
  <div class="info">
    <div class="n">{name1} vs {name2}</div>
    <div class="l">{group} · {kickoff}</div>
  </div>
  <div class="pct">
    <div class="v">{name1} {b[0]}%</div>
    <div class="label">平 {b[1]}% · {name2} {b[2]}%</div>
    <div class="progress-bar">
      <div class="seg sh" style="width:{hw}%"></div>
      <div class="seg sd" style="width:{dw}%"></div>
      <div class="seg sa" style="width:{aw}%"></div>
    </div>
  </div>
</a>""")

    return idx_template.replace("{date}", datetime.now().strftime("%m/%d %H:00")).replace("{items}", "\n".join(items))


# ============ MAIN ============

def main():
    matches = {m["slug"]: m for m in load_matches()}
    poly_all = load_poly_data()
    today = get_today_slugs()

    if not today:
        print("ℹ️  没有当天比赛数据")
        return

    results = []
    os.makedirs(os.path.join(REPO_DIR, "match"), exist_ok=True)

    for slug in today:
        if slug not in matches:
            print(f"  ⚠️  {slug} 不在 matches.json 中，跳过")
            continue

        match = matches[slug]
        poly = poly_all.get(slug)

        print(f"\n{'='*40}")
        print(f"{match['flag1']} {match['name1']} vs {match['flag2']} {match['name2']}")

        # Run model
        try:
            probs, expected, elos, raw = run_model(match["team1"], match["team2"])
            if probs[0] is None:
                print(f"  ❌ 模型解析失败")
                print(raw)
                continue
            m_h, m_d, m_a = probs
            e_h, e_a = expected or (0, 0)
            elo_h, elo_a = elos or (0, 0)
            print(f"  📊 模型: {m_h:.1f}% / {m_d:.1f}% / {m_a:.1f}%")
            model_probs = (round(m_h, 1), round(m_d, 1), round(m_a, 1))
        except Exception as ex:
            print(f"  ❌ 模型: {ex}")
            continue

        # Polymarket
        poly_probs = None
        if poly:
            p_h = poly.get("home_prob")
            p_d = poly.get("draw_prob")
            p_a = poly.get("away_prob")
            if all(v is not None for v in [p_h, p_d, p_a]):
                poly_probs = (p_h, p_d, p_a)
                print(f"  🎲 Polymarket: {p_h}% / {p_d}% / {p_a}%")
            else:
                print(f"  🎲 Polymarket: 数据不完整")
        else:
            print(f"  🎲 Polymarket: N/A")

        # Blend
        blended = blend((m_h, m_d, m_a), poly_probs)
        b_h, b_d, b_a = blended
        print(f"  ✅ 融合(α={ALPHA}): {b_h}% / {b_d}% / {b_a}%")

        entry = {
            "match": match,
            "poly": poly_probs,
            "model": model_probs,
            "blended": blended,
            "expected": (round(e_h, 2), round(e_a, 2)),
            "elos": (elo_h, elo_a),
        }
        results.append(entry)

        # Generate detail page
        html = generate_match_html(match, poly_probs, model_probs, (e_h, e_a), blended, (elo_h, elo_a))
        filepath = os.path.join(REPO_DIR, "match", f"{slug}.html")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  ✅ 已更新 match/{slug}.html")

    # Update index page
    if results:
        index_html = update_list_page(results)
        with open(os.path.join(REPO_DIR, "index.html"), "w", encoding="utf-8") as f:
            f.write(index_html)
        print(f"\n✅ 已更新 index.html ({len(results)} 场比赛)")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'比赛':<16} {'模型':<18} {'Poly':<18} {'融合':<18}")
    print(f"{'-'*60}")
    for r in results:
        name = f"{r['match']['flag1']}{r['match']['name1']}vs{r['match']['flag2']}{r['match']['name2']}"
        m = f"{r['model'][0]}%/{r['model'][1]}%/{r['model'][2]}%"
        p = f"{r['poly'][0]}%/{r['poly'][1]}%/{r['poly'][2]}%" if r['poly'] else "N/A"
        b = f"{r['blended'][0]}%/{r['blended'][1]}%/{r['blended'][2]}%"
        print(f"{name:<16} {m:<18} {p:<18} {b:<18}")


if __name__ == "__main__":
    main()
