import os, re, json, uuid, hmac, csv, io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, select, desc
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from questions import Q

load_dotenv()
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./data/rayebod.db")
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DB_URL.startswith("postgresql://"):
    DB_URL = DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {}, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Assessment(Base):
    __tablename__ = "investor_assessments"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    investor_name = Column(String(200), nullable=False)
    risk_level = Column(String(50), nullable=False)
    risk_score = Column(Integer, nullable=False)
    investor_profile_name = Column(String(250), nullable=False)
    report = Column(Text, nullable=False)
    data_json = Column(Text, nullable=False)

Base.metadata.create_all(bind=engine)

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GENERIC_LLM_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_KEY = GEMINI_KEY or GENERIC_LLM_KEY
# Configure Gemini and OpenAI-compatible providers independently so an example
# GEMINI_MODEL value cannot accidentally override an OpenRouter model ID.
if GEMINI_KEY:
    LLM_BASE = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
else:
    LLM_BASE = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash:free")
LLM_LAST_ERROR = ""
ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "").strip()
DIMS = {"tol": "تحمل ریسک", "cap": "توان مالی پذیرش ریسک", "kno": "دانش سرمایه‌گذاری", "hor": "افق و هدف"}
SESSIONS: dict[str, dict] = {}
app = FastAPI(title="Rayebod Investor Profiler")
security = HTTPBasic()

SYS = ("شما دستیار حرفه‌ای شناخت مشتری در شرکت سبدگردان هستید. به فارسی رسمی، روشن و بی‌طرف پاسخ دهید. "
       "توصیه خریدوفروش، پیش‌بینی بازار یا تضمین بازده ندهید. فقط بر اساس داده‌های ارائه‌شده تحلیل کنید؛ "
       "اگر اطلاعاتی موجود نیست، آن را حدس نزنید و برای بررسی انسانی علامت بزنید.")

class ChatMessage(BaseModel):
    sid: Optional[str] = None
    message: str = Field(default="", max_length=4000)
    option: Optional[str] = None

class AdminUpdate(BaseModel):
    report: str = Field(min_length=1, max_length=20000)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if not ADMIN_PASS:
        raise HTTPException(status_code=503, detail="برای فعال‌کردن بک‌آفیس، متغیر ADMIN_PASSWORD را تنظیم کنید")
    user_ok = hmac.compare_digest(credentials.username.encode(), ADMIN_USER.encode())
    pass_ok = hmac.compare_digest(credentials.password.encode(), ADMIN_PASS.encode())
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="نام کاربری یا رمز عبور نادرست است", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

async def llm(messages, temperature=0.25):
    global LLM_LAST_ERROR
    if not LLM_KEY:
        LLM_LAST_ERROR = "کلید هوش مصنوعی تنظیم نشده است."
        return None
    headers = {"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"}
    if "openrouter.ai" in LLM_BASE.lower():
        headers["X-Title"] = "Rayebod Investor Profiler"
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
    try:
        async with httpx.AsyncClient(timeout=35) as client:
            response = await client.post(
                f"{LLM_BASE}/chat/completions",
                headers=headers,
                json={"model": LLM_MODEL, "temperature": temperature,
                      "messages": [{"role": "system", "content": SYS}] + messages},
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                LLM_LAST_ERROR = "سرویس هوش مصنوعی پاسخ متنی خالی برگرداند."
                return None
            LLM_LAST_ERROR = ""
            return content.strip()
    except httpx.HTTPStatusError as exc:
        # Keep only a short provider response for the authenticated diagnostic route.
        body = exc.response.text[:500]
        LLM_LAST_ERROR = f"خطای سرویس هوش مصنوعی (HTTP {exc.response.status_code}): {body}"
        return None
    except Exception as exc:
        LLM_LAST_ERROR = f"خطا در اتصال به سرویس هوش مصنوعی: {type(exc).__name__}"
        return None

def opts_of(q):
    return [{"key": k, "text": t} for k, t, _ in q.get("opts", [])]
FA = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

async def ask(i):
    # پرسش‌ها ثابت می‌مانند تا معنا و امتیازدهی در هر جلسه پایدار باشد.
    return Q[i]["text"]

async def classify(q, text):
    normalized = text.strip().casefold()
    for key, label, _score in q.get("opts", []):
        if normalized == label.strip().casefold() or normalized == key.casefold():
            return key, None
    if not LLM_KEY:
        return None, "برای جلوگیری از ثبت پاسخ اشتباه، لطفاً یکی از گزینه‌های مشخص‌شده را انتخاب کنید."
    choices = "\n".join(f"{k}: {t}" for k, t, _ in q["opts"])
    out = await llm([{"role": "user", "content": f"سؤال: {q['text']}\nگزینه‌ها:\n{choices}\nپاسخ کاربر: {text}\nفقط JSON معتبر بده: {{\"key\":\"حرف گزینه یا null\",\"reply\":\"در صورت ابهام، پرسش روشن‌کنندهٔ کوتاه\"}}"}], 0)
    m = re.search(r"\{.*\}", out or "", re.S)
    try:
        d = json.loads(m.group(0))
        key = d.get("key")
        return (key if key in [o[0] for o in q["opts"]] else None), d.get("reply")
    except Exception:
        return None, None

PERSONAS = {
 "income": {"name":"درآمدمحور و نقدشوندگی‌طلب", "description":"برای شما دسترسی به وجه و برنامه‌ریزی جریان نقدی اهمیت ویژه‌ای دارد.", "strengths":"توجه به نیازهای کوتاه‌مدت و جریان نقدی", "watch":"نیاز به روشن‌کردن زمان و میزان برداشت‌های احتمالی"},
 "family": {"name":"خانواده‌محور و آینده‌نگر", "description":"تصمیم‌های مالی شما بهتر است با اهداف خانوادگی و تعهدات پیش‌رو هماهنگ باشد.", "strengths":"توجه به اهداف بلندمدت خانواده و تعهدات", "watch":"تفکیک سرمایهٔ اهداف نزدیک از سرمایهٔ بلندمدت"},
 "entrepreneur": {"name":"کارآفرین با درآمد متغیر", "description":"به‌دلیل متغیر بودن درآمد، مدیریت نقدینگی و جداسازی دارایی شخصی از کسب‌وکار برای شما اهمیت دارد.", "strengths":"انعطاف‌پذیری و آشنایی با عدم‌قطعیت", "watch":"تعیین ذخیرهٔ اضطراری و جلوگیری از تمرکز ریسک"},
 "wealth": {"name":"دارایی‌محور و تنوع‌طلب", "description":"به نظر می‌رسد حفظ و سازمان‌دهی دارایی در کنار رشد و تنوع‌بخشی برای شما مهم است.", "strengths":"نگاه کل‌نگر به دارایی و امکان تنوع‌بخشی", "watch":"بررسی تمرکز دارایی‌ها، نقدشوندگی و افق هر هدف"},
 "preserver": {"name":"حافظ ارزش و امنیت‌محور", "description":"حفظ اصل سرمایه و قابل‌پیش‌بینی بودن مسیر سرمایه‌گذاری برای شما اولویت بالاتری دارد.", "strengths":"توجه به پایداری و محدودیت زیان", "watch":"بررسی تعادل میان حفظ ارزش واقعی دارایی و ریسک‌پذیری"},
 "planner": {"name":"برنامه‌ریز متعادل", "description":"رویکرد شما بیشتر با برنامه‌ریزی مرحله‌ای و سنجش تعادل میان اهداف و نوسان سازگار است.", "strengths":"توجه به هدف، زمان و توازن تصمیم‌ها", "watch":"تبدیل اهداف کلی به افق‌ها و نیازهای مالی مشخص"},
 "grower": {"name":"رشدگرا و آینده‌ساز", "description":"رشد سرمایه در افق بلندمدت یکی از انگیزه‌های اصلی شما به نظر می‌رسد.", "strengths":"تمرکز بر رشد و اهداف بلندمدت", "watch":"سنجش ظرفیت واقعی زیان و پرهیز از تمرکز بیش از حد"},
 "active": {"name":"فرصت‌جوی فعال و تحلیل‌محور", "description":"پاسخ‌های شما می‌تواند با علاقه به فرصت‌های متنوع و مشارکت فعال در تصمیم‌گیری سازگار باشد.", "strengths":"علاقه به یادگیری و مشارکت فعال", "watch":"تفکیک اعتمادبه‌نفس از دانش اثبات‌شده و کنترل رفتار هیجانی"},
}

def profile_for(ans, risk, dims):
    # Profile/personality is selected independently from the three-level risk category.
    goal = ans.get("q21")
    fam, job, wealth = ans.get("fam"), ans.get("job"), ans.get("wealth")
    horizon = next((q for q in Q if q["id"] == "q20"), None)
    horizon_score = next((s for k, _, s in horizon["opts"] if k == ans.get("q20")), 50) if horizon else 50
    income_dependency = next((s for k, _, s in next(q for q in Q if q["id"]=="q9")["opts"] if k == ans.get("q9")), 50)
    if income_dependency <= 35 or horizon_score <= 40 or goal == "B": key = "income"
    elif fam in ("C", "D") and goal != "D": key = "family"
    elif job == "C": key = "entrepreneur"
    elif wealth == "B": key = "wealth"
    elif risk == "کم‌ریسک" or goal == "A": key = "preserver"
    elif risk == "پرریسک" and dims.get("دانش سرمایه‌گذاری", 0) >= 60 and horizon_score >= 70: key = "active"
    elif goal == "D" and horizon_score >= 70: key = "grower"
    else: key = "planner"
    persona = PERSONAS[key]
    return {"id": key, "name": persona["name"], "description": persona["description"],
            "risk_level": risk, "strengths": persona["strengths"], "watch": persona["watch"]}

def score(ans):
    acc = {d: [] for d in DIMS}
    for q in Q:
        if q.get("dim") in DIMS and q["id"] in ans and q.get("opts"):
            found = next((s for k, _, s in q["opts"] if k == ans[q["id"]]), None)
            if found is not None: acc[q["dim"]].append(found)
    dims_raw = {d: round(sum(v) / len(v)) if v else 0 for d, v in acc.items()}
    total = 0.50*dims_raw["tol"] + 0.25*dims_raw["cap"] + 0.15*dims_raw["hor"] + 0.10*dims_raw["kno"]
    # Conservative guardrails: short horizon / low capacity should trigger human review.
    flags = []
    if dims_raw["cap"] < 30:
        total = min(total, 39)
        flags.append("توان مالی پذیرش ریسک پایین است؛ سطح ریسک بالاتر بدون بررسی انسانی تأیید نشود.")
    if dims_raw["hor"] < 30 and dims_raw["tol"] >= 65:
        flags.append("ناهمخوانی احتمالی میان تحمل ریسک بالا و افق کوتاه سرمایه‌گذاری.")
    risk = "کم‌ریسک" if total < 40 else "ریسک متوسط" if total < 65 else "پرریسک"
    dims = {DIMS[d]: v for d, v in dims_raw.items()}
    p = profile_for(ans, risk, dims)
    return {"dims": dims, "risk_score": round(total), "risk_level": risk, "age": ans.get("age"),
            "investor_profile": p, "review_flags": flags}

def fallback_report(name, result, open_answers):
    p = result["investor_profile"]
    flags = "\n".join("- " + f for f in result.get("review_flags", [])) or "- در پاسخ‌های امتیازدهی‌شده، هشدار خودکار مشخصی ثبت نشده است؛ بررسی انسانی همچنان لازم است."
    open_text = "\n".join(f"- {k}: {v}" for k, v in open_answers.items()) or "پاسخ تشریحی ثبت نشده است."
    return (f"خلاصه ارزیابی: {name} در این ارزیابی امتیاز ریسک {result['risk_score']} از ۱۰۰ و طبقهٔ «{result['risk_level']}» دارد. "
            f"تیپ سرمایه‌گذار پیشنهادی برای شناخت بهتر او «{p['name']}» است؛ این تیپ شخصیت مالی/نیازها را توصیف می‌کند و با سطح ریسک یکی نیست.\n\n"
            f"برداشت اولیه: {p['description']} نقطهٔ قوت احتمالی: {p['strengths']}.\n\n"
            f"موارد نیازمند توجه: {p['watch']}\n{flags}\n\n"
            f"پاسخ‌های تشریحی برای گفت‌وگوی مدیر سبد:\n{open_text}\n\n"
            "این گزارش مقدماتی است؛ پیش از تصمیم یا پیشنهاد محصول، اطلاعات مالی، نیاز نقدینگی و تناسب سرمایه‌گذاری باید توسط کارشناس بررسی شود.")

async def create_report(name, result, answers):
    open_answers = {q["text"]: answers[q["id"]] for q in Q if q.get("kind") == "text" and q["id"] in answers}
    payload = {"result": result, "open_answers": open_answers, "answers": answers}  # نام مشتری عمداً به سرویس AI ارسال نمی‌شود
    report = await llm([{"role":"user", "content":
       "یک گزارش فارسی رسمی برای مدیر سبد، حداکثر ۵ بند کوتاه، بنویس. حتماً این دو خروجی را جدا و روشن نگه دار: (۱) پروفایل ریسک فقط یکی از سه سطح کم‌ریسک، ریسک متوسط، پرریسک است؛ (۲) پروفایل سرمایه‌گذار یک تیپ شخصیتی/نیازی است و می‌تواند بین افراد هم‌سطح ریسک متفاوت باشد. از داده‌های ارائه‌شده فراتر نرو. پاسخ‌های تشریحی را خلاصه کن، تناقض‌های احتمالی و پرسش‌های پیگیری را ذکر کن. توصیهٔ خریدوفروش نده. اگر اطلاعات کافی نیست صریح بگو.\n" + json.dumps(payload, ensure_ascii=False)}], 0.2)
    return report or fallback_report(name, result, open_answers)

@app.get("/api/health")
def health():
    return {"ok": True, "ai_configured": bool(LLM_KEY), "ai_provider": "Gemini" if GEMINI_KEY else ("OpenAI-compatible" if GENERIC_LLM_KEY else "not configured"),
            "ai_model": LLM_MODEL if LLM_KEY else None,
            "database": "postgresql" if DB_URL.startswith("postgres") else "sqlite"}

@app.post("/api/ai-test")
async def ai_test(_admin=Depends(require_admin)):
    if not LLM_KEY:
        raise HTTPException(status_code=400, detail="کلید هوش مصنوعی تنظیم نشده است.")
    result = await llm([{"role": "user", "content": "فقط عبارت READY را برگردان."}], 0)
    if not result:
        raise HTTPException(status_code=502, detail=LLM_LAST_ERROR or "پاسخ معتبر از سرویس هوش مصنوعی دریافت نشد.")
    return {"ok": True, "provider": "Gemini" if GEMINI_KEY else "OpenAI-compatible", "model": LLM_MODEL, "sample": result[:200]}

@app.post("/api/chat")
async def chat(m: ChatMessage, db: Session = Depends(get_db)):
    if not m.sid or m.sid not in SESSIONS:
        sid = str(uuid.uuid4())
        SESSIONS[sid] = {"stage":"name", "i":0, "ans":{}, "name":""}
        return {"sid": sid, "reply":"سلام و وقت‌بخیر. برای شروع، لطفاً نام و نام خانوادگی خود را وارد بفرمایید.", "options":[], "progress":0, "stage":"name"}
    s = SESSIONS[m.sid]
    if s["stage"] == "name":
        name = m.message.strip()
        if len(name) < 2 or len(name) > 120:
            return {"sid":m.sid, "reply":"لطفاً نام خود را وارد بفرمایید (حداقل دو نویسه).", "options":[], "progress":0, "stage":"name"}
        s["name"] = name
        s["stage"] = "questions"
        greet = f"ممنون، {name}. این پرسشنامه برای شناخت اولیهٔ وضعیت مالی، تحمل ریسک و تیپ سرمایه‌گذاری شماست. لطفاً پاسخ‌ها را با توجه به شرایط واقعی خود انتخاب کنید."
        return {"sid":m.sid, "reply":greet+"\n\n"+await ask(0), "options":opts_of(Q[0]), "progress":0, "stage":"questions"}
    if s["stage"] == "done":
        return {"sid":m.sid, "reply":"این ارزیابی تکمیل شده است. برای شروع ارزیابی تازه، صفحه را دوباره بارگذاری کنید.", "options":[], "progress":1}
    q = Q[s["i"]]
    if q.get("kind") == "number":
        match = re.search(r"\d+", m.message.translate(FA)); key = int(match.group()) if match else None
        ok = key is not None and 15 <= key <= 100
        note = "لطفاً سن خود را به‌صورت عدد بین ۱۵ تا ۱۰۰ وارد بفرمایید."
    elif q.get("kind") == "text":
        key = m.message.strip(); ok = len(key) >= 8
        note = "لطفاً کمی بیشتر توضیح بفرمایید؛ یا اگر موردی ندارید، همین را کوتاه بنویسید."
    else:
        key, note = (m.option, None) if m.option else await classify(q, m.message)
        ok = key in [o[0] for o in q["opts"]]
        if not ok and not note: note = "لطفاً یکی از گزینه‌ها را انتخاب کنید؛ برای ثبت دقیق پاسخ از گزینه‌های مشخص‌شده استفاده کنید."
    if not ok:
        return {"sid":m.sid, "reply":note, "options":opts_of(q), "progress":s["i"]/len(Q), "stage":"questions"}
    s["ans"][q["id"]] = key
    s["i"] += 1
    if s["i"] < len(Q):
        return {"sid":m.sid, "reply":await ask(s["i"]), "options":opts_of(Q[s["i"]]), "progress":s["i"]/len(Q), "stage":"questions"}
    result = score(s["ans"])
    report = await create_report(s["name"], result, s["ans"])
    open_answers = {q["text"]: s["ans"][q["id"]] for q in Q if q.get("kind")=="text" and q["id"] in s["ans"]}
    record_data = {"answers":s["ans"], "result":result, "open_answers":open_answers}
    rec = Assessment(investor_name=s["name"], risk_level=result["risk_level"], risk_score=result["risk_score"],
                     investor_profile_name=result["investor_profile"]["name"], report=report,
                     data_json=json.dumps(record_data, ensure_ascii=False))
    db.add(rec); db.commit(); db.refresh(rec)
    s["stage"] = "done"
    return {"sid":m.sid, "reply":report, "options":[], "progress":1, "result":result, "record_id":rec.id, "stage":"done"}

@app.get("/api/admin/records")
def admin_records(q: str = "", risk: str = "", limit: int = 200, _admin=Depends(require_admin), db: Session = Depends(get_db)):
    stmt = select(Assessment).order_by(desc(Assessment.created_at)).limit(min(max(limit,1),500))
    rows = db.scalars(stmt).all()
    if q.strip():
        term = q.strip().casefold()
        rows = [r for r in rows if term in r.investor_name.casefold() or term in r.investor_profile_name.casefold() or term in r.report.casefold()]
    if risk: rows = [r for r in rows if r.risk_level == risk]
    return [{"id":r.id,"created_at":r.created_at.isoformat(),"name":r.investor_name,"risk_level":r.risk_level,
             "risk_score":r.risk_score,"profile":r.investor_profile_name} for r in rows]

@app.get("/api/admin/records/{record_id}")
def admin_record(record_id: int, _admin=Depends(require_admin), db: Session = Depends(get_db)):
    r = db.get(Assessment, record_id)
    if not r: raise HTTPException(404, "پرونده پیدا نشد")
    return {"id":r.id,"created_at":r.created_at.isoformat(),"name":r.investor_name,"risk_level":r.risk_level,
            "risk_score":r.risk_score,"profile":r.investor_profile_name,"report":r.report,"data":json.loads(r.data_json)}

@app.put("/api/admin/records/{record_id}/report")
def admin_update_report(record_id: int, body: AdminUpdate, _admin=Depends(require_admin), db: Session = Depends(get_db)):
    r = db.get(Assessment, record_id)
    if not r: raise HTTPException(404, "پرونده پیدا نشد")
    r.report = body.report.strip(); db.commit()
    return {"ok":True}

@app.get("/api/admin/export.csv")
def admin_export(_admin=Depends(require_admin), db: Session = Depends(get_db)):
    rows = db.scalars(select(Assessment).order_by(desc(Assessment.created_at))).all()
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["id","created_at","investor_name","risk_level","risk_score","investor_profile_name","report"])
    for r in rows: writer.writerow([r.id,r.created_at.isoformat(),r.investor_name,r.risk_level,r.risk_score,r.investor_profile_name,r.report])
    data = stream.getvalue().encode("utf-8-sig")
    return StreamingResponse(io.BytesIO(data), media_type="text/csv; charset=utf-8", headers={"Content-Disposition":"attachment; filename=rayebod_assessments.csv"})

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return (ROOT / "frontend" / "admin.html").read_text(encoding="utf-8")

app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="static")
