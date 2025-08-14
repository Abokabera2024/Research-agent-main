# Research-agent
Scientific agent

مخطط تنفيذ تفصيلي (Runbook) لبناء Agent بحثي يقرأ التوثيق، يحلّل علميًا بـ SciPy، ويتخذ قرارات باستخدام LangGraph

> هذه خطة تنفيذ عملية خطوة-بخطوة، مع هيكل مجلدات، أوامر تثبيت، ونماذج كود جاهزة للانطلاق. صيغتها قابلة للنسخ والبدء فورًا.




---

0) الهدف والنتيجة المتوقعة

الهدف: وكيل/Agent بحثي يقوم تلقائيًا بـ:

1. قراءة ملفات PDF/مستندات التوثيق.


2. استخراج مقاطع مهمة (نص/معادلات/جداول).


3. إجراء تحليلات علمية/إحصائية بـ SciPy عند الحاجة.


4. اتخاذ قرار (مثل: الورقة ذات صلة؟ النتائج صحيحة إحصائيًا؟)


5. إنتاج تقرير مُنظَّم + تخزين الملخصات والنتائج للرجوع لاحقًا.



النتيجة: Pipeline مستدام بذاكرة وحالة، يُعاد تشغيله تلقائيًا عند إضافة ملفات جديدة، ويمكن استدعاؤه من CLI أو API.


---

1) بيئة العمل وهيكل المشروع

1.1 إنشاء المشروع وتفعيل البيئة

mkdir research-agent
cd research-agent
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

1.2 المتطلبات (requirements.txt)

أنشئ ملف requirements.txt بالمحتوى التالي:

langchain>=0.2
langgraph>=0.2
langchain-openai>=0.1
chromadb>=0.5
sentence-transformers>=3.0
pypdf>=4.2
unstructured>=0.15
scipy>=1.12
numpy>=1.26
pydantic>=2.7
typer>=0.12
fastapi>=0.111
uvicorn>=0.30
structlog>=24.1
python-dotenv>=1.0

ثم:

pip install -r requirements.txt

> ملاحظة: لو عايز نموذج محلي بدل مزود سحابي، أضف: ollama>=0.3 واستخدمه بدل OpenAI (اختياري).



1.3 هيكل المجلدات

research-agent/
├─ .venv/
├─ .env                      # مفاتيح API وإعدادات
├─ requirements.txt
├─ data/
│  ├─ inbox/                 # ملفات PDF الجديدة (تلقائيًا تُعالَج)
│  ├─ processed/             # ملفات تم معالجتها
│  └─ examples/              # أمثلة
├─ storage/
│  ├─ vectors/               # قاعدة Chroma
│  └─ checkpoints/           # حفظ حالة LangGraph (SQLite)
├─ src/
│  ├─ config.py
│  ├─ schema.py              # تعريف State وModels
│  ├─ loaders.py             # قراءة واستخراج نصوص من PDF
│  ├─ chunking.py            # تقطيع النصوص
│  ├─ embeddings.py          # إنشاء المتجهات/الفهرسة
│  ├─ tools_scipy.py         # وظائف SciPy التحليلية
│  ├─ nodes.py               # عقد LangGraph (ingest/analyze/decide/report)
│  ├─ graph.py               # بناء وربط الرسم البياني
│  ├─ reporter.py            # توليد التقارير
│  ├─ run_cli.py             # واجهة سطر أوامر
│  └─ api.py                 # واجهة REST (FastAPI)
└─ README.md


---

2) الإعدادات المفاتيح (ملف .env)

أنشئ .env بالمحتوى المناسب:

OPENAI_API_KEY=sk-...              # أو بدائله إن تستخدم مزودًا آخر
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini              # غيّر للاسم الذي تفضّله/المتاح لك
CHROMA_DIR=./storage/vectors
CHECKPOINT_DB=./storage/checkpoints/graph_state.sqlite

> لو هتستخدم نموذج محلي (Ollama مثلاً):
LLM_MODEL=llama3 واستخدم واجهة langchain_community.llms المخصصة.




---

3) تعريف الـ State ونماذج البيانات (src/schema.py)

from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class DocChunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    meta: Dict[str, Any] = {}

class AnalysisResult(BaseModel):
    findings: List[str] = []
    stats: Dict[str, Any] = {}
    needs_scipy: bool = False
    rationale: str = ""

class Decision(BaseModel):
    label: str               # e.g., "relevant", "irrelevant", "uncertain"
    confidence: float
    criteria: List[str] = []

class Report(BaseModel):
    summary: str
    methods: List[str]
    decisions: Decision
    attachments: Dict[str, Any] = {}

# حالة الرسم البياني:
from typing import TypedDict

class GraphState(TypedDict, total=False):
    doc_path: str
    doc_id: str
    raw_text: str
    chunks: List[DocChunk]
    query: str
    retrieved: List[DocChunk]
    analysis: AnalysisResult
    scipy_out: Dict[str, Any]
    decision: Decision
    report: Report
    human_feedback: Optional[str]


---

4) التحميل والاستخراج (src/loaders.py)

from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_pdf_text(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    # دمج النصوص
    text = "\n".join([p.page_content for p in pages])
    return text

def assign_doc_id(pdf_path: str) -> str:
    p = Path(pdf_path)
    return f"{p.stem}"


---

5) التقطيع/التقسيم (src/chunking.py)

from typing import List
from .schema import DocChunk
import textwrap
import uuid

def simple_chunk(text: str, doc_id: str, chunk_size: int = 1200, overlap: int = 150) -> List[DocChunk]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        segment = text[start:end]
        chunk_id = f"{doc_id}-{uuid.uuid4().hex[:8]}"
        chunks.append(DocChunk(doc_id=doc_id, chunk_id=chunk_id, text=segment))
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


---

6) التضمين والفهرسة (src/embeddings.py)

from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from .schema import DocChunk
import os

def get_embeddings():
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)

def build_or_load_vectorstore(chunks: List[DocChunk], persist_dir: str):
    embeddings = get_embeddings()
    texts = [c.text for c in chunks]
    metadatas = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id} for c in chunks]
    vs = Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas, persist_directory=persist_dir)
    vs.persist()
    return vs

def as_retriever(persist_dir: str, k: int = 5):
    embeddings = get_embeddings()
    vs = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    return vs.as_retriever(search_kwargs={"k": k})

> بديل محلي بدون API: استخدم sentence-transformers (مثل all-MiniLM-L6-v2) مع FAISS بدل Chroma.




---

7) أدوات التحليل العلمي بـ SciPy (src/tools_scipy.py)

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats, optimize

def ttest_from_text(numbers_a: List[float], numbers_b: List[float]) -> Dict[str, Any]:
    t, p = stats.ttest_ind(numbers_a, numbers_b, equal_var=False, nan_policy="omit")
    return {"test": "t_independent", "t": float(t), "p": float(p)}

def curve_fit_example(x: List[float], y: List[float]) -> Dict[str, Any]:
    # مثال بسيط y = a * x + b
    def model(x, a, b):
        return a * x + b
    popt, pcov = optimize.curve_fit(model, np.array(x), np.array(y))
    a, b = popt
    return {"model": "y=a*x+b", "a": float(a), "b": float(b)}

def decide_need_scipy(text: str) -> bool:
    # معيار مبسّط: لو النص يحوي "p-value" أو "regression" أو بيانات أرقام
    triggers = ["p-value", "regression", "ANOVA", "t-test", "significance"]
    return any(t.lower() in text.lower() for t in triggers)

> يمكنك استبدال استخراج الأرقام بـ Regex/Parser مخصّص حسب مجال بحثك (قياسات، أزمنة، أطوال موجية…).




---

8) عقد LangGraph (src/nodes.py)

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .schema import GraphState, DocChunk, AnalysisResult, Decision, Report
from .loaders import load_pdf_text, assign_doc_id
from .chunking import simple_chunk
from .embeddings import build_or_load_vectorstore, as_retriever
from .tools_scipy import ttest_from_text, curve_fit_example, decide_need_scipy
import os
import re
import json

def llm():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)

def node_ingest(state: GraphState) -> GraphState:
    path = state["doc_path"]
    doc_id = assign_doc_id(path)
    raw = load_pdf_text(path)
    state.update({"doc_id": doc_id, "raw_text": raw})
    return state

def node_split_embed(state: GraphState) -> GraphState:
    chunks = simple_chunk(state["raw_text"], doc_id=state["doc_id"])
    state["chunks"] = chunks
    vs = build_or_load_vectorstore(chunks, persist_dir=os.getenv("CHROMA_DIR", "./storage/vectors"))
    return state

def node_retrieve(state: GraphState) -> GraphState:
    retriever = as_retriever(persist_dir=os.getenv("CHROMA_DIR", "./storage/vectors"), k=6)
    q = state.get("query") or "Extract the study aims, methods, data, and key results."
    docs = retriever.get_relevant_documents(q)
    retrieved = []
    for i, d in enumerate(docs):
        retrieved.append(DocChunk(doc_id=d.metadata.get("doc_id",""), chunk_id=d.metadata.get("chunk_id",""), text=d.page_content))
    state["retrieved"] = retrieved
    return state

def node_analyze(state: GraphState) -> GraphState:
    llm_model = llm()
    content = "\n\n".join([c.text for c in state["retrieved"]])[:12000]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a scientific research assistant. Be precise and concise."),
        ("user", """Analyze the following scientific text. 
- Extract aims, methodology, datasets, equations, and any referenced statistics.
- Indicate whether SciPy calculations are required to verify or extend results.
- Return JSON with keys: findings(list of strings), stats(dict), needs_scipy(bool), rationale(str).
TEXT:
{content}""")
    ])
    msg = prompt.format_messages(content=content)
    out = llm_model.invoke(msg)
    try:
        data = json.loads(out.content)
    except Exception:
        # fallback parsing
        data = {"findings": [out.content[:8000]], "stats": {}, "needs_scipy": decide_need_scipy(content), "rationale": "LLM fallback"}
    state["analysis"] = AnalysisResult(**data)
    return state

def _extract_number_lists(text: str) -> List[float]:
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return [float(n) for n in nums[:50]]

def node_scipy_compute(state: GraphState) -> GraphState:
    # مثال: إن وُجدت قائمتان من الأرقام في المقاطع المسترجعة، نجرب t-test/curve_fit
    merged = "\n".join(c.text for c in state.get("retrieved", []))
    nums = _extract_number_lists(merged)
    scipy_out = {}
    if len(nums) >= 10:
        mid = len(nums)//2
        scipy_out["ttest"] = ttest_from_text(nums[:mid], nums[mid:])
        # محاولة ملائمة خطية بسيطة
        x = list(range(len(nums)))
        scipy_out["curve_fit"] = curve_fit_example(x[:len(nums)], nums)
    state["scipy_out"] = scipy_out
    return state

def node_decide(state: GraphState) -> GraphState:
    llm_model = llm()
    analysis = state["analysis"]
    scipy_out = state.get("scipy_out", {})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior researcher. Decide relevance and rigor."),
        ("user", f"""Based on the analysis and SciPy results, decide if this document is RELEVANT to the research topic.
Return JSON: label("relevant"/"irrelevant"/"uncertain"), confidence(0-1), criteria(list).
ANALYSIS: {analysis.model_dump()}
SCIPY: {json.dumps(scipy_out)[:4000]}""")
    ])
    out = llm_model.invoke(prompt.format_messages())
    try:
        data = json.loads(out.content)
    except Exception:
        data = {"label":"uncertain","confidence":0.5,"criteria":["fallback"]}
    state["decision"] = Decision(**data)
    return state

def node_report(state: GraphState) -> GraphState:
    analysis = state["analysis"]
    decision = state["decision"]
    txt = [
        f"# Research Agent Report for: {state['doc_id']}",
        "## Summary of Findings",
        *[f"- {f}" for f in analysis.findings[:10]],
        "## Statistical Checks",
        f"- Stats: {json.dumps(analysis.stats)[:1200]}",
        f"- SciPy: {json.dumps(state.get('scipy_out', {}))[:1200]}",
        "## Decision",
        f"- Label: {decision.label} (confidence={decision.confidence:.2f})",
        f"- Criteria: {', '.join(decision.criteria[:6])}",
    ]
    state["report"] = Report(
        summary="\n".join(txt),
        methods=["RAG (Chroma)","LLM analysis","SciPy checks"],
        decisions=decision,
        attachments={}
    )
    return state


---

9) بناء الرسم البياني (src/graph.py)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from .schema import GraphState
from .nodes import (
    node_ingest, node_split_embed, node_retrieve,
    node_analyze, node_scipy_compute, node_decide, node_report
)
import os

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("ingest", node_ingest)
    builder.add_node("split_embed", node_split_embed)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("analyze", node_analyze)
    builder.add_node("scipy_compute", node_scipy_compute)
    builder.add_node("decide", node_decide)
    builder.add_node("report", node_report)

    builder.set_entry_point("ingest")
    builder.add_edge("ingest", "split_embed")
    builder.add_edge("split_embed", "retrieve")
    builder.add_edge("retrieve", "analyze")

    # تفرّع شرطي: إن كانت الحاجة لـ SciPy
    def route_scipy(state: GraphState):
        if state.get("analysis") and state["analysis"].needs_scipy:
            return "scipy_compute"
        return "decide"

    builder.add_conditional_edges("analyze", route_scipy, {
        "scipy_compute": "scipy_compute",
        "decide": "decide"
    })

    builder.add_edge("scipy_compute", "decide")
    builder.add_edge("decide", "report")
    builder.add_edge("report", END)

    cp = SqliteSaver.from_conn_string(os.getenv("CHECKPOINT_DB", "./storage/checkpoints/graph_state.sqlite"))
    return builder.compile(checkpointer=cp)


---

10) توليد التقارير وحفظ النتائج (src/reporter.py)

from pathlib import Path
from datetime import datetime

def save_report(markdown_text: str, out_dir: str, doc_id: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(out_dir) / f"{doc_id}-{ts}.md"
    path.write_text(markdown_text, encoding="utf-8")
    return str(path)


---

11) تشغيل من سطر الأوامر (src/run_cli.py)

import os
import typer
from dotenv import load_dotenv
from .graph import build_graph
from .reporter import save_report

app = typer.Typer()

@app.command()
def run(pdf_path: str, query: str = typer.Option(None, help="Optional retrieval query")):
    load_dotenv()
    graph = build_graph()
    config = {"configurable": {"thread_id": f"run-{os.path.basename(pdf_path)}"}}
    state = {"doc_path": pdf_path}
    if query:
        state["query"] = query

    # stream للأحداث (اختياري) أو invoke مباشر:
    final = graph.invoke(state, config)
    report = final["report"].summary
    out = save_report(report, "./storage/reports", final["doc_id"])
    typer.echo(f"Report saved: {out}")

if __name__ == "__main__":
    app()

التشغيل:

python -m src.run_cli --help
python -m src.run_cli run data/examples/paper.pdf --query "Model specification, dataset sizes, and p-values"


---

12) API للنشر (src/api.py)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
from .graph import build_graph
from .reporter import save_report
import shutil
import os

app = FastAPI()
graph = None

@app.on_event("startup")
def _init():
    global graph
    graph = build_graph()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), query: str = Form(None)):
    inbox = Path("data/inbox")
    inbox.mkdir(parents=True, exist_ok=True)
    dst = inbox / file.filename
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    config = {"configurable": {"thread_id": f"api-{file.filename}"}}
    init_state = {"doc_path": str(dst)}
    if query:
        init_state["query"] = query

    final = graph.invoke(init_state, config)
    report = final["report"].summary
    out = save_report(report, "./storage/reports", final["doc_id"])
    return JSONResponse({"report_path": out, "decision": final["decision"].model_dump()})

التشغيل:

uvicorn src.api:app --reload --port 8000
# أرسل ملف PDF عبر POST إلى /analyze


---

13) تشغيل تلقائي عند إضافة ملفات (اختياري – Watcher بسيط)

استخدم watchdog (اختياري):

pip install watchdog

سكربت بسيط يراقب data/inbox/ ويشغل CLI تلقائيًا لكل ملف جديد.


---

14) إدخال البشر في الحلقة (Human-in-the-Loop) — خيار سريع

بسيط: بعد توليد القرار، علّق التنفيذ إن كانت الثقة < 0.7 واطلب موافقة بشرية قبل حفظ التقرير النهائي.

إضافة منطق في node_decide:

لو confidence < 0.7 خزّن التقرير كـ draft واطلب مراجعة (أضف حقل needs_review=True في Report.attachments).


واجهة مراجعة: صفحة بسيطة (FastAPI + HTML) تعرض الملخص وأزرار Approve/Reject تُعيد استدعاء Endpoint لتحديث human_feedback وإعادة استكمال الرسم البياني (يمكنك تشغيل graph.invoke مرة أخرى مع الحالة المحدثة).



---

15) التسجيل والمراقبة (Observability)

استخدم structlog لتسجيل منظم JSON.

فعّل سجلات كل Node (بداية/نهاية/مدة/مخرجات مختصرة).

(اختياري) استخدم LangSmith/Callbacks إن متاح.


مثال بسيط:

import structlog
log = structlog.get_logger()

# داخل كل node_*:
log.info("node_start", node="analyze", doc_id=state.get("doc_id"))
# ...
log.info("node_end", node="analyze", decision=getattr(state.get("decision",""), "label", None))


---

16) التحقق والاختبارات (QA)

اختبارات وحدة:

تغذية عينة PDF معروفة + توقع قرار محدد.

اختبار مسار “بدون SciPy” ومسار “مع SciPy”.


قياسات:

نسبة التطابق مع قرارات بشرية سابقة.

زمن المعالجة/التكلفة.

دقة الاسترجاع (Recall@k) باختبار أسئلة معيارية.



استخدم pytest وشغّل:

pip install pytest
pytest -q


---

17) الحاوية (Docker) — اختياري

Dockerfile مبسّط:

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

بناء وتشغيل:

docker build -t research-agent:latest .
docker run -p 8000:8000 --env-file .env -v $(pwd)/storage:/app/storage -v $(pwd)/data:/app/data research-agent:latest


---

18) نصائح عملية للتخصيص بمجالك البحثي

1. محولات خاصة لاستخراج الأرقام/الجداول:

طوّر Parser يلتقط الجداول (مثل قياسات/عوائد/دقة تصنيف) ويحفظها بصيغة pandas.DataFrame ثم تمريرها لـ SciPy.



2. قواعد قرار مخصّصة:

حدّد معايير قبول (مثل p < 0.05، حجم عينة ≥ N، وجود تكرار تجريبي) وادمجها في node_decide.



3. ذاكرة موضوعية:

خصص فهارس Chroma لكل مشروع/موضوع للحد من “التلوث” بين المواضيع.



4. تحكم في التكلفة:

اضبط temperature=0 وقلّل حجم النص الممرَّر للـ LLM (Trim + Summaries).



5. أمان:

لا تنفّذ أي كود “مقتبس من الورقة” بدون Sandbox.

راقب حجم المدخلات لمنع الهجمات (PDFs ضخمة/معيبة).





---

19) تجربة سريعة (Proof-of-Concept)

1. ضع ملفًا تجريبيًا في data/examples/paper.pdf.


2. شغّل:



python -m src.run_cli run data/examples/paper.pdf --query "What are the study aims and statistical tests?"

3. افتح التقرير الناتج داخل storage/reports/*.md.




---

ختام

بهذا المخطط التنفيذي ستملك Agent بحثي متكامل: قراءة → استرجاع → تحليل LLM → تحقّق SciPy → قرار → تقرير، مع ذاكرة وحفظ حالة تسمح بالاستئناف والتوسع لاحقًا (CLI أو API).
