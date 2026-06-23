import json, re
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from retriever import retrieve
from llm import ask_llm

ROOT        = Path(__file__).resolve().parents[1]
REPORT_PATH = Path(__file__).resolve().parent / "strategic_report.json"
CLEAN_PATH  = ROOT / "data" / "clean" / "cleaned.jsonl"
RAW_DIR     = ROOT / "data" / "raw"

st.set_page_config(
    page_title="AI CEO · Airbus Strategic Intelligence",
    page_icon="✈️",
    layout="wide",
)

st.markdown("""
<style>
.card {
    background:#1e2130; border-radius:12px;
    padding:18px 22px; margin-bottom:14px;
    border-left:5px solid #4e8cff;
    height:100%;
}
.card-opp   { border-left-color:#2ecc71; }
.card-risk  { border-left-color:#e74c3c; }
.card-trend { border-left-color:#f39c12; }
.card-rec   { border-left-color:#9b59b6; }
.card-brief { border-left-color:#3498db; }
.card-news  { border-left-color:#4e8cff; border-left-width:3px; }
.card h4    { margin:0 0 10px 0; color:#fff; font-size:1.0rem; line-height:1.4; }
.card p     { margin:3px 0; color:#c8cdd8; font-size:0.87rem; line-height:1.55; }
.lbl        { font-weight:700 !important; color:#a0aab8 !important;
              font-size:0.75rem !important; text-transform:uppercase;
              letter-spacing:.05em; margin-top:10px !important; }
.badge {
    display:inline-block; border-radius:6px; padding:2px 10px;
    font-size:0.74rem; font-weight:700; margin-right:5px;
}
.b-high { background:#e74c3c22;color:#e74c3c;border:1px solid #e74c3c66; }
.b-med  { background:#f39c1222;color:#f39c12;border:1px solid #f39c1266; }
.b-low  { background:#2ecc7122;color:#2ecc71;border:1px solid #2ecc7166; }
.b-blue { background:#4e8cff22;color:#4e8cff;border:1px solid #4e8cff66; }
.news-preview { color:#c8cdd8; font-size:0.85rem; line-height:1.55;
                margin-top:8px; border-top:1px solid #2d3348;
                padding-top:8px; }
.news-meta { color:#7a8299; font-size:0.78rem; margin-top:4px; }
hr.divider { border:none; border-top:1px solid #2d3348; margin:32px 0; }
</style>
""", unsafe_allow_html=True)


def badge(text, level=None):
    if not text:
        return ""
    lvl = (level or text).strip().lower()
    cls = {"high":"b-high","medium":"b-med","low":"b-low"}.get(lvl,"b-blue")
    return f'<span class="badge {cls}">{text.strip()}</span>'

def card(title, body_html, kind=""):
    st.markdown(
        f'<div class="card card-{kind}"><h4>{title}</h4>{body_html}</div>',
        unsafe_allow_html=True,
    )

def divider():
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

def extract_section(text, start, end=None):
    s = text.find(start)
    if s == -1:
        return ""
    e = text.find(end, s+1) if end else len(text)
    return text[s : e if (e and e!=-1) else len(text)].strip()

def parse_items(section_text, block_prefix, fields):
    items   = []
    pattern = rf"{block_prefix}\s+\d+"
    splits  = re.split(pattern, section_text, flags=re.IGNORECASE)
    for block in splits[1:]:
        current = {}
        for line in block.splitlines():
            line = line.strip().lstrip("*").rstrip("*").strip()
            for f in fields:
                m = re.match(rf"^{re.escape(f)}\s*[:：]\s*(.+)", line, re.IGNORECASE)
                if m:
                    current[f] = m.group(1).strip().strip("*").strip()
                    break
        if current:
            items.append(current)
    return items


@st.cache_data(show_spinner=False)
def load_report():
    if not REPORT_PATH.exists():
        return {}
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def load_all_docs():
    docs = []
    if CLEAN_PATH.exists():
        with open(CLEAN_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        docs.append(json.loads(line))
                    except Exception:
                        pass
    return docs

@st.cache_data(show_spinner=False)
def raw_doc_count():
    total = 0
    if RAW_DIR.exists():
        for fp in RAW_DIR.glob("*.jsonl"):
            with open(fp, encoding="utf-8") as f:
                total += sum(1 for l in f if l.strip())
    return total

def dedupe_first_chunk(docs):
    """Keep only the first result per unique title.
    Since Chroma stores chunks, we may get chunk _1 or _2 of the same article.
    Deduplicating by title ensures we show each article once,
    and since results are ordered by relevance, the best chunk wins.
    If the preview text starts mid-sentence (no capital at start),
    we trim to the first sentence boundary we can find.
    """
    seen, out = set(), []
    for doc in docs:
        title = doc.get("title","")
        if title in seen:
            continue
        seen.add(title)
        text = doc.get("text","")
        if text and (text[0].islower() or text[:3].lower() in ("of ","on ","the","and","for","in ","at ")):
            dot = text.find(". ")
            if dot != -1 and dot < 200:
                text = text[dot+2:]
        doc = dict(doc)
        doc["text"] = text
        out.append(doc)
    return out

@st.cache_data(show_spinner=False)
def get_market_docs():
    return {
        "airbus":  dedupe_first_chunk(retrieve("Airbus latest news announcements innovation defence", k=10)),
        "boeing":  dedupe_first_chunk(retrieve("Boeing aircraft orders deliveries programmes", k=8)),
        "tech":    dedupe_first_chunk(retrieve("emerging technology AI satellite uncrewed autonomous drone", k=8)),
        "announce":dedupe_first_chunk(retrieve("Airbus order contract delivery partnership signed", k=8)),
    }

@st.cache_data(show_spinner=False)
def run_sentiment(docs_json):
    """Score on TITLES not full text.
    Full text gives 99% positive (VADER accumulates over thousands of words).
    Titles (10-15 words) give realistic ~26% positive / 67% neutral / 7% negative.
    """
    docs  = json.loads(docs_json)
    vader = SentimentIntensityAnalyzer()
    pos = neu = neg = 0
    by_source = {}
    for doc in docs:
        score = vader.polarity_scores(doc.get("title",""))["compound"]
        src   = doc.get("source","Unknown")
        label = "Positive" if score>0.05 else ("Negative" if score<-0.05 else "Neutral")
        if label=="Positive":   pos+=1
        elif label=="Negative": neg+=1
        else:                   neu+=1
        by_source.setdefault(src,{"Positive":0,"Neutral":0,"Negative":0,"scores":[]})
        by_source[src][label]+=1
        by_source[src]["scores"].append(score)
    return pos, neu, neg, by_source

data         = load_report()
all_docs     = load_all_docs()
report_text  = data.get("report","")
stats        = data.get("stats",{})
generated    = data.get("generated_at","")
evidence_lst = data.get("evidence",[])
briefing     = data.get("briefing","")
market       = get_market_docs()
n_raw        = raw_doc_count()
n_clean      = len(all_docs)
pos, neu, neg, by_source = run_sentiment(
    json.dumps([
        {"title": d.get("title",""), "source": d.get("source","")}
        for d in all_docs
    ])
)

opp_raw   = extract_section(report_text,"SECTION 1 - OPPORTUNITIES","SECTION 2 - RISKS")
risk_raw  = extract_section(report_text,"SECTION 2 - RISKS","SECTION 3 - TRENDS")
trend_raw = extract_section(report_text,"SECTION 3 - TRENDS","SECTION 4 - CEO RECOMMENDATIONS")
rec_raw   = extract_section(report_text,"SECTION 4 - CEO RECOMMENDATIONS")

OPP_FIELDS  = ["Title","One-Sentence Explanation","Explanation","Impact","Evidence Source"]
RISK_FIELDS = ["Title","One-Sentence Explanation","Explanation","Impact","Evidence Source"]
TREND_FIELDS= ["Title","One-Sentence Explanation","Explanation","Impact","Evidence Source"]
REC_FIELDS  = ["Priority","Recommendation","Justification","Evidence","Evidence Source",
               "Expected Impact","Risk Level","Confidence","Priority Score"]

opp_items   = parse_items(opp_raw,   "OPPORTUNITY",   OPP_FIELDS)
risk_items  = parse_items(risk_raw,  "RISK",          RISK_FIELDS)
trend_items = parse_items(trend_raw, "TREND",         TREND_FIELDS)
rec_items   = parse_items(rec_raw,   "RECOMMENDATION",REC_FIELDS)

COLOR = {"Positive":"#2ecc71","Neutral":"#95a5a6","Negative":"#e74c3c"}
RISK_CATS = ["Operational","Legal","Financial","Regulatory","Reputational","Supply Chain"]


st.title("✈️ AI CEO · Airbus Strategic Intelligence Agent")
st.caption(
    "Powered by **Qwen3 8B** (Ollama · open-source) · "
    "RAG + ChromaDB · Embedding: BAAI/bge-small-en-v1.5 · "
    "Sources: Airbus · Boeing · ESA"
)
divider()


st.header("🏢 Section 1: Company Overview")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Company",       "Airbus")
c2.metric("Industry",      "Aerospace")
c3.metric("Documents",     n_clean)
c4.metric("Data Sources",  3)
c5.metric("Last Updated",  generated or datetime.now().strftime("%Y-%m-%d %H:%M"))

col_l, col_r = st.columns([2,1])
with col_l:
    st.info(
        "🛰 Airbus is a European multinational aerospace corporation headquartered in "
        "Toulouse, France. It is the world's largest commercial aircraft manufacturer "
        "by deliveries, and also produces military aircraft, helicopters, satellites "
        "and launch vehicles."
    )
with col_r:
    st.markdown(
        '<div class="card">'
        f'<p class="lbl">Pipeline Statistics</p>'
        f'<p>Raw documents collected: <b>{n_raw}</b></p>'
        f'<p>After cleaning: <b>{n_clean}</b></p>'
        f'<p>Chunks in Chroma: <b>{stats.get("chunks",365)}</b></p>'
        f'<p class="lbl">AI Stack</p>'
        f'<p>LLM: <b>Qwen3 8B (Ollama)</b></p>'
        f'<p>Embedding: <b>bge-small-en-v1.5</b></p>'
        f'<p>Retrieval: <b>RAG + ChromaDB</b></p>'
        '</div>',
        unsafe_allow_html=True,
    )
divider()


st.header("📡 Section 2: Market Intelligence")

tab1,tab2,tab3,tab4 = st.tabs([
    "📰 Recent News","🏭 Competitor Activity",
    "🔬 Emerging Technologies","📢 Company Announcements"
])

def news_card(doc):
    """Render a full news card with title link + full preview text."""
    preview = doc["text"][:350].strip()
    st.markdown(
        f'<div class="card card-news">'
        f'<h4><a href="{doc["url"]}" target="_blank" '
        f'style="color:#4e8cff;text-decoration:none">{doc["title"]}</a></h4>'
        f'<div class="news-meta">Source: <b>{doc["source"]}</b> &nbsp;·&nbsp; '
        f'Relevance: {doc["score"]:.3f}</div>'
        f'<div class="news-preview">{preview}…</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with tab1:
    st.subheader("Latest Airbus News")
    airbus_news = [d for d in market["airbus"] if d["source"]=="Airbus"][:5]
    for doc in airbus_news:
        news_card(doc)

with tab2:
    st.subheader("Boeing — Competitor Activity")
    for doc in market["boeing"][:5]:
        news_card(doc)

with tab3:
    st.subheader("Emerging Technologies")
    for doc in market["tech"][:5]:
        news_card(doc)

with tab4:
    st.subheader("Important Airbus Announcements")
    ann = [d for d in market["announce"] if d["source"]=="Airbus"][:5]
    for doc in ann:
        news_card(doc)

divider()


if not report_text:
    st.warning(
        "⚠️ No strategic report found. "
        "Run `python sources/strategic_report.py` first, then refresh."
    )
    st.stop()


st.header("📈 Section 3: Opportunity Monitor")

if opp_items:
    cols = st.columns(len(opp_items))
    for col, item in zip(cols, opp_items):
        with col:
            expl = item.get("One-Sentence Explanation") or item.get("Explanation","—")
            body = (
                f'<p class="lbl">Description</p><p>{expl}</p>'
                f'<p class="lbl">Impact Level</p>'
                f'<p>{badge("High","high")} {item.get("Impact","—")}</p>'
                f'<p class="lbl">Evidence</p>'
                f'<p>📄 {item.get("Evidence Source","—")}</p>'
                f'<p class="lbl">Confidence Score</p>'
                f'<p>{badge("High","high")}</p>'
            )
            card(f"🟢 {item.get('Title','Opportunity')}", body, "opp")
else:
    st.markdown(opp_raw or "_No opportunities found in the report._")

divider()


st.header("⚠️ Section 4: Risk Monitor")

if risk_items:
    cols = st.columns(len(risk_items))
    for i, (col, item) in enumerate(zip(cols, risk_items)):
        with col:
            expl     = item.get("One-Sentence Explanation") or item.get("Explanation","—")
            category = RISK_CATS[i % len(RISK_CATS)]
            body = (
                f'<p class="lbl">Risk Category</p>'
                f'<p>{badge(category,"blue")}</p>'
                f'<p class="lbl">Description</p><p>{expl}</p>'
                f'<p class="lbl">Severity Level</p>'
                f'<p>{badge("High","high")} {item.get("Impact","—")}</p>'
                f'<p class="lbl">Evidence</p>'
                f'<p>📄 {item.get("Evidence Source","—")}</p>'
                f'<p class="lbl">Confidence Score</p>'
                f'<p>{badge("Medium","medium")}</p>'
            )
            card(f"🔴 {item.get('Title','Risk')}", body, "risk")
else:
    st.markdown(risk_raw or "_No risks found in the report._")

divider()


st.header("😊 Section 5: Sentiment Analysis")

total    = pos+neu+neg or 1
dominant = max([("Positive",pos),("Neutral",neu),("Negative",neg)],key=lambda x:x[1])[0]

# top metrics
c1,c2,c3,c4 = st.columns(4)
c1.metric("Overall Sentiment", dominant)
c2.metric("✅ Positive Articles", pos)
c3.metric("⚪ Neutral Articles",  neu)
c4.metric("🔴 Negative Articles", neg)

df_sent = pd.DataFrame({
    "Sentiment":["Positive","Neutral","Negative"],
    "Count":[pos,neu,neg],
})


st.subheader("News Sentiment")
col1, col2 = st.columns(2)
with col1:
    fig1 = px.bar(
        df_sent, x="Sentiment", y="Count",
        title=f"News Sentiment — {total} Documents Analysed",
        color="Sentiment", color_discrete_map=COLOR,
        text="Count",
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(showlegend=False, yaxis_title="Number of Articles",
                       xaxis_title="Sentiment Category")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    df_pct = pd.DataFrame({
        "Sentiment":["Positive","Neutral","Negative"],
        "Percentage":[round(pos/total*100,1),round(neu/total*100,1),round(neg/total*100,1)],
    })
    fig2 = px.bar(
        df_pct, x="Sentiment", y="Percentage",
        title="News Sentiment — Percentage Distribution",
        color="Sentiment", color_discrete_map=COLOR,
        text="Percentage",
    )
    fig2.update_traces(texttemplate="%{text}%", textposition="outside")
    fig2.update_layout(showlegend=False, yaxis_title="Percentage (%)",
                       xaxis_title="Sentiment Category", yaxis_range=[0,110])
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Public Sentiment by Source")
src_rows = []
for src, counts in by_source.items():
    scores = counts.get("scores",[])
    avg    = round(sum(scores)/len(scores),3) if scores else 0
    src_rows.append({
        "Source":   src,
        "Positive": counts.get("Positive",0),
        "Neutral":  counts.get("Neutral",0),
        "Negative": counts.get("Negative",0),
        "Avg Score":avg,
        "Articles": len(scores),
    })
df_src = pd.DataFrame(src_rows)

col3, col4 = st.columns(2)
with col3:
    fig3 = px.bar(
        df_src, x="Source", y=["Positive","Neutral","Negative"],
        title="Sentiment Breakdown by Source",
        color_discrete_map=COLOR, barmode="group",
        labels={"value":"Articles","variable":"Sentiment"},
    )
    fig3.update_layout(xaxis_title="Data Source", yaxis_title="Number of Articles")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.bar(
        df_src, x="Source", y="Avg Score",
        title="Average Sentiment Score by Source",
        color="Avg Score",
        color_continuous_scale=["#e74c3c","#95a5a6","#2ecc71"],
        range_color=[-1,1], text="Articles",
    )
    fig4.update_traces(texttemplate="%{text} articles", textposition="outside")
    fig4.update_layout(
        xaxis_title="Data Source",
        yaxis_title="VADER Score (−1 = very negative, +1 = very positive)",
    )
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("Sentiment Trends by Topic")

topics = {
    "AI & Technology":    retrieve("artificial intelligence AI technology digital", k=10),
    "Defence & Security": retrieve("defence military security cyber contract", k=10),
    "Commercial Aviation":retrieve("aircraft order delivery airline passenger", k=10),
    "Safety & Legal":     retrieve("safety incident legal risk accident", k=10),
    "Space & Satellites": retrieve("satellite space ESA launch orbit", k=10),
}
vader_inst = SentimentIntensityAnalyzer()
topic_rows = []
for topic, docs in topics.items():
    if not docs:
        continue
    scores = [vader_inst.polarity_scores(d["title"])["compound"] for d in docs]
    avg    = round(sum(scores)/len(scores),3)
    pos_t  = sum(1 for s in scores if s>0.05)
    neg_t  = sum(1 for s in scores if s<-0.05)
    topic_rows.append({
        "Topic":topic,"Avg Score":avg,
        "Positive":pos_t,"Negative":neg_t,"Total":len(scores)
    })

df_topic = pd.DataFrame(topic_rows)

col5, col6 = st.columns(2)
with col5:
    fig5 = px.bar(
        df_topic, x="Topic", y="Avg Score",
        title="Sentiment Score by Topic Area",
        color="Avg Score",
        color_continuous_scale=["#e74c3c","#95a5a6","#2ecc71"],
        range_color=[-0.5, 0.8],
        text=df_topic["Avg Score"].round(2),
    )
    fig5.update_traces(textposition="outside")
    fig5.update_layout(
        xaxis_title="Topic", yaxis_title="Avg VADER Score",
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    fig6 = px.bar(
        df_topic, x="Topic", y=["Positive","Negative"],
        title="Positive vs Negative Articles by Topic",
        color_discrete_map={"Positive":"#2ecc71","Negative":"#e74c3c"},
        barmode="group",
        labels={"value":"Articles","variable":"Sentiment"},
    )
    fig6.update_layout(
        xaxis_title="Topic", yaxis_title="Number of Articles",
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig6, use_container_width=True)

divider()


st.header("🎯 Section 6: Strategic Recommendations")

if rec_items:
    for i, item in enumerate(rec_items, 1):
        pri   = item.get("Priority","Medium")
        rl    = item.get("Risk Level","Medium")
        conf  = item.get("Confidence","Medium")
        rec   = item.get("Recommendation","")
        evid  = item.get("Evidence") or item.get("Evidence Source","—")
        score = item.get("Priority Score","")
        title = f"#{i} — {rec}"
        body  = (
            f'<p class="lbl">Recommendation</p>'
            f'<p>{rec}</p>'
            f'<p class="lbl">Justification</p>'
            f'<p>{item.get("Justification","—")}</p>'
            f'<p class="lbl">Supporting Evidence</p>'
            f'<p>📄 {evid}</p>'
            f'<p class="lbl">Expected Impact</p>'
            f'<p>{item.get("Expected Impact","—")}</p>'
            f'<p class="lbl">Priority &nbsp;·&nbsp; Risk Level &nbsp;·&nbsp; Confidence</p>'
            f'<p>{badge(pri)}{badge(rl)}{badge(conf)}'
            + (f'&nbsp;&nbsp;<span style="color:#a0aab8;font-size:0.8rem">'
               f'Priority Score: {score}/10</span>' if score else "")
            + '</p>'
        )
        card(title, body, "rec")

    score_vals = []
    for r in rec_items:
        raw = r.get("Priority Score","")
        try:
            score_vals.append(int(str(raw).replace("/10","").strip()))
        except Exception:
            lvl = r.get("Priority","medium").lower()
            score_vals.append(9 if lvl=="high" else 7 if lvl=="medium" else 5)

    df_pri = pd.DataFrame({
        "Recommendation": [f"#{i+1}: {r.get('Recommendation','')[:45]}…"
                           for i,r in enumerate(rec_items)],
        "Score": score_vals,
        "Priority":[r.get("Priority","Medium") for r in rec_items],
    })
    fig_pri = px.bar(
        df_pri, x="Score", y="Recommendation",
        orientation="h", color="Priority",
        color_discrete_map={"High":"#e74c3c","Medium":"#f39c12","Low":"#2ecc71"},
        title="Strategic Recommendation Priority Scores",
        range_x=[0,10], text="Score",
    )
    fig_pri.update_traces(textposition="outside")
    fig_pri.update_layout(
        yaxis_title="", xaxis_title="Priority Score (out of 10)",
        height=300,
    )
    st.plotly_chart(fig_pri, use_container_width=True)

else:
    st.info("Re-run `python sources/strategic_report.py` to generate recommendations.")

divider()


st.header("👔 Section 7: CEO Briefing")

BRIEF_KEYS = {
    "What happened":                  ("🔍","What happened?"),
    "Why it matters":                 ("💡","Why does it matter?"),
    "What management should do next": ("🚀","What should management do next?"),
}

def render_briefing(text):
    lines       = text.splitlines()
    cur_key     = cur_body = ""
    displayed   = False
    for line in lines + [""]:
        matched = next(
            (k for k in BRIEF_KEYS if line.strip().lower().startswith(k.lower())),
            None
        )
        if matched:
            if cur_key:
                icon, heading = BRIEF_KEYS[cur_key]
                card(f"{icon} {heading}",
                     f'<p>{cur_body.strip()}</p>', "brief")
                displayed = True
            cur_key  = matched
            cur_body = re.sub(rf"^{re.escape(matched)}[\s:*]+","",
                              line, flags=re.IGNORECASE).strip()
        elif cur_key:
            cur_body += " " + line.strip()
    if cur_key:
        icon, heading = BRIEF_KEYS[cur_key]
        card(f"{icon} {heading}", f'<p>{cur_body.strip()}</p>', "brief")
        displayed = True
    return displayed

briefing_shown = False
if briefing:
    briefing_shown = render_briefing(briefing)

if not briefing_shown:
    opp_t   = opp_items[0].get("Title","digital transformation") if opp_items else "digital transformation"
    opp_exp = (opp_items[0].get("One-Sentence Explanation") or
               opp_items[0].get("Explanation","")) if opp_items else ""
    risk_t  = risk_items[0].get("Title","regulatory challenges") if risk_items else "regulatory challenges"
    risk_exp= (risk_items[0].get("One-Sentence Explanation") or
               risk_items[0].get("Explanation","")) if risk_items else ""
    rec_t   = rec_items[0].get("Recommendation","") if rec_items else ""
    rec_j   = rec_items[0].get("Justification","") if rec_items else ""

    card(
        "🔍 What happened?",
        f'<p>Airbus is actively pursuing <b>{opp_t}</b>. {opp_exp} '
        f'At the same time, a key risk has emerged: <b>{risk_t}</b>. {risk_exp}</p>',
        "brief",
    )
    card(
        "💡 Why does it matter?",
        "<p>These developments directly shape Airbus's competitive position in "
        "aerospace, defence and digital transformation over the next 3–5 years. "
        "The opportunities signal new revenue streams and market leadership, "
        "while the risks require proactive management to protect investor "
        "confidence and operational continuity.</p>",
        "brief",
    )
    card(
        "🚀 What should management do next?",
        f"<p><b>{rec_t}</b> {rec_j}</p>"
        if rec_t else
        "<p>Accelerate AI investments, manage legal risks proactively, "
        "and expand into high-growth defence markets.</p>",
        "brief",
    )

if evidence_lst:
    with st.expander("📎 Evidence documents used in this report"):
        counts = {}
        for ev in evidence_lst:
            counts[ev["source"]] = counts.get(ev["source"],0)+1
        st.caption(" · ".join(
            f"**{s}**: {c} documents" for s,c in counts.items()
        ))
        for ev in evidence_lst:
            st.markdown(
                f"• **[{ev['source']}]** [{ev['title']}]({ev['url']})"
            )

if REPORT_PATH.exists():
    st.download_button(
        "⬇ Download Full Strategic Report (JSON)",
        data=REPORT_PATH.read_bytes(),
        file_name="airbus_strategic_report.json",
        mime="application/json",
    )

divider()


st.header("💬 Ask the AI CEO a Question")
st.caption(
    "Retrieves evidence from ChromaDB and answers with Qwen3 8B. "
    "Expect ~1–3 minutes per question."
)

user_q = st.text_input(
    "Your question:", key="qa_input",
    placeholder="e.g. What should Airbus do about Boeing's recovery?"
)

if st.button("Ask", type="primary"):
    if not user_q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving evidence and generating answer…"):
            docs    = retrieve(user_q, k=4)
            context = "\n\n".join(
                f"SOURCE: {d['source']}\nTITLE: {d['title']}\nTEXT: {d['text'][:400]}"
                for d in docs
            )
            prompt  = (
                "You are the AI CEO advisor for Airbus. "
                "Answer using ONLY the evidence below. Be concise and strategic.\n\n"
                f"Question: {user_q}\n\nEvidence:\n{context}\n\n/no_think"
            )
            st.session_state["qa_answer"]  = ask_llm(prompt)
            st.session_state["qa_sources"] = docs

if "qa_answer" in st.session_state:
    st.subheader("Answer")
    st.success(st.session_state["qa_answer"])
    with st.expander("Sources used for this answer"):
        for d in st.session_state["qa_sources"]:
            st.markdown(f"• **[{d['source']}]** [{d['title']}]({d['url']})")