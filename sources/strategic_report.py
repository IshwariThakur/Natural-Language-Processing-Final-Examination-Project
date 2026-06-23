import json, time
from datetime import datetime
from pathlib import Path
from retriever import retrieve, get_collection
from llm import ask_llm

REPORT_PATH  = Path(__file__).resolve().parent / "strategic_report.json"
EVIDENCE_K    = 3      
EVIDENCE_CHARS = 300   

QUERIES = [
    "Airbus opportunities new markets partnerships emerging technology",
    "Airbus risks safety legal regulatory financial",
    "Airbus trends digital AI sustainability defence",
    "Boeing competitor aircraft orders deliveries",
]


def gather_evidence():
    seen, docs = set(), []
    for q in QUERIES:
        for d in retrieve(q, k=EVIDENCE_K):
            if d["title"] not in seen:
                seen.add(d["title"])
                docs.append(d)
    return docs


def fmt(evidence):
    return "\n\n".join(
        f"EVIDENCE_SOURCE: {d['title']}\nSOURCE: {d['source']}\n"
        f"TEXT: {d['text'][:EVIDENCE_CHARS]}"
        for d in evidence
    )


def step1_intelligence(context):
    prompt = f"""You are a Strategic Intelligence Analyst for Airbus.
Use ONLY the evidence below. No outside knowledge.
Every item MUST cite an Evidence Source copied from an EVIDENCE_SOURCE line.

SECTION 1 - OPPORTUNITIES
OPPORTUNITY 1
Title:
Explanation:
Impact:
Evidence Source:

OPPORTUNITY 2
Title:
Explanation:
Impact:
Evidence Source:

SECTION 2 - RISKS
RISK 1
Title:
Explanation:
Impact:
Evidence Source:

RISK 2
Title:
Explanation:
Impact:
Evidence Source:

SECTION 3 - TRENDS
TREND 1
Title:
Explanation:
Impact:
Evidence Source:

TREND 2
Title:
Explanation:
Impact:
Evidence Source:

EVIDENCE:
{context}

/no_think"""
    return ask_llm(prompt)


def step2_recommendations(intelligence):
    intel_short = intelligence[:600]
    prompt = f"""You are the CEO of Airbus. Write 3 strategic recommendations. Fill every field. Be concise.

SECTION 4 - CEO RECOMMENDATIONS

RECOMMENDATION 1
Priority:
Recommendation:
Justification:
Expected Impact:
Risk Level:
Confidence:
Evidence Source:

RECOMMENDATION 2
Priority:
Recommendation:
Justification:
Expected Impact:
Risk Level:
Confidence:
Evidence Source:

RECOMMENDATION 3
Priority:
Recommendation:
Justification:
Expected Impact:
Risk Level:
Confidence:
Evidence Source:

INTELLIGENCE SUMMARY:
{intel_short}

/no_think"""
    return ask_llm(prompt)


def step3_briefing(report_text):
    prompt = f"""Write a short CEO briefing using ONLY the report below.
One short paragraph for each heading:

What happened:
Why it matters:
What management should do next:

REPORT:
{report_text[:1200]}

/no_think"""
    return ask_llm(prompt)


def get_stats():
    try:
        col    = get_collection()
        chunks = col.count()
        metas  = col.get(include=["metadatas"]).get("metadatas", [])
        docs   = len({m.get("parent_id") for m in metas if m})
    except Exception:
        chunks, docs = 0, 0
    return {"documents": docs, "chunks": chunks, "sources": 3}


def build_report():
    print("Gathering evidence...")
    evidence = gather_evidence()
    print(f"  {len(evidence)} unique documents")
    context  = fmt(evidence)

    print("Step 1/3 — intelligence (opportunities / risks / trends)...")
    intelligence = step1_intelligence(context)

    print("Step 2/3 — CEO recommendations...")
    recommendations = step2_recommendations(intelligence)

    report_text = intelligence.strip() + "\n\n" + recommendations.strip()

    print("Step 3/3 — executive briefing...")
    briefing = step3_briefing(report_text)

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "stats":    get_stats(),
        "report":   report_text,
        "briefing": briefing.strip(),
        "evidence": [
            {"title": d["title"], "source": d["source"], "url": d["url"]}
            for d in evidence
        ],
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    return data


def generate_report():
    return build_report()["report"]


if __name__ == "__main__":
    t0   = time.time()
    data = build_report()
    print(f"\n{'='*60}")
    print("STRATEGIC REPORT SAVED")
    print(f"{'='*60}")
    print(f"File:      {REPORT_PATH}")
    print(f"Documents: {data['stats']['documents']}   Chunks: {data['stats']['chunks']}")
    print(f"Time:      {time.time()-t0:.1f}s")

    report = data["report"]
    for s in ["SECTION 1","SECTION 2","SECTION 3","SECTION 4"]:
        status = "✓" if s in report else "✗ MISSING"
        print(f"  {status}  {s}")
    print(f"\nNext: streamlit run streamlit_app.py")