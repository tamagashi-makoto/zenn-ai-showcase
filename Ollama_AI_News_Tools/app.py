import os  # ← 最優先で環境変数を設定するために先に読み込む

# ====== APIキーを“import ollama”の前に設定する（重要） ======
DEFAULT_OLLAMA_API_KEY = "PUT_YOUR_OLLAMA_API"
_env_key = (os.getenv("OLLAMA_API_KEY") or DEFAULT_OLLAMA_API_KEY).strip()
# 非ASCII対策（念のため）
try:
    if any(ord(ch) > 127 for ch in _env_key):
        _env_key = DEFAULT_OLLAMA_API_KEY
except Exception:
    pass
os.environ["OLLAMA_API_KEY"] = _env_key  # ← ここで確実に設定してから ollama を import

import streamlit as st
import textwrap
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import requests
import ollama
import urllib.parse
import xml.etree.ElementTree as ET
import html

# ==============================
# ページ設定とスタイル（UIの基本はそのまま）
# ==============================
st.set_page_config(
    page_title="AI News Daily",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    .subtitle { text-align: center; color: #6c757d; font-size: 1.2rem; margin-bottom: 2rem; }
    .stButton>button {
        width: 100%; background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: 600; border-radius: 8px;
        padding: 0.75rem 1.5rem; border: none; margin-top: 0.5rem;
    }
    .section-title {
        font-size: 1.4rem; font-weight: 700; margin: 1rem 0 0.5rem 0;
    }
    .source-card {
        padding: 0.75rem 1rem; border: 1px solid #eaeaea; border-radius: 8px;
        background: #fafafa; margin-bottom: 0.75rem;
    }
    .footer {
        color: #6c757d; font-size: 0.9rem; text-align: center; margin-top: 2rem;
    }
    a { text-decoration: none; font-size: 0.9rem; }
    /* 入力窓を大きく見やすく */
    .big-chat textarea {
        min-height: 160px !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# 検索クライアント（公式仕様準拠）
# 1) REST:   POST https://ollama.com/api/web_search  Authorization: Bearer <OLLAMA_API_KEY>
#    body:   {"query": "..."}
# 2) Python: ollama.web_search(query)
# ==============================
class UniversalSearchClient:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        self.api_key = (api_key or os.getenv("OLLAMA_API_KEY", "")).strip()
        self.timeout = timeout

        # 公式RESTの固定エンドポイント
        self.fixed_search_url = (os.getenv("OLLAMA_WEB_SEARCH_URL", "")).strip() or "https://ollama.com/api/web_search"

        # HTTPセッション（Authorization: Bearer のみ）
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
            })
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; AI-News-Dashboard/1.0)",
        })

    # -------- 1) 公式RESTエンドポイント（先に叩く：上限エラーを即検出） --------
    def _try_ollama_http(self, query: str, max_results: int) -> Optional[List[Dict]]:
        try:
            resp = self.session.post(
                self.fixed_search_url,
                json={"query": query},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                items = self._normalize_search(data)[:max_results]
                if not items:
                    st.info("[REST] 200 だが results が空。クエリ内容/対象期間の可能性。")
                else:
                    st.info(f"[REST] web_search OK: {len(items)} 件")
                return items if items else None
            else:
                snippet = (resp.text or "")[:400].replace("\n"," ")
                st.info(f"[REST] {resp.status_code} {self.fixed_search_url}  body={snippet}")
                # 402/401/403/429 はここで打ち切る（SDKで二重に叩かない）
                if resp.status_code in (401, 402, 403, 429):
                    return None
        except Exception as e:
            st.info(f"[REST] 例外: {e}")
        return None

    # -------- 2) Ollama Python SDK --------
    def _try_ollama_sdk(self, query: str, max_results: int) -> Optional[List[Dict]]:
        try:
            res = ollama.web_search(query)
            items = self._normalize_search(res)[:max_results]
            if not items:
                st.info("[SDK] web_search は成功しましたが、results が空でした。")
            else:
                st.info(f"[SDK] web_search OK: {len(items)} 件")
            return items if items else None
        except Exception as e:
            st.info(f"[SDK] web_search 例外: {e}")
            return None

    # -------- 公開：検索（REST→SDK の順で試す） --------
    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        res = self._try_ollama_http(query, max_results)
        if res:
            return res
        res = self._try_ollama_sdk(query, max_results)
        if res:
            return res
        st.info("SDK/REST いずれも結果ゼロでした。APIキー・レート・ネットワークをご確認ください。")
        return []

    # -------- 正規化（{"results":[...]} 想定） --------
    @staticmethod
    def _normalize_search(data) -> List[Dict]:
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if isinstance(data.get("results"), list):
                items = data["results"]
            else:
                items = next((v for v in data.values() if isinstance(v, list)), [])
        else:
            items = []
        norm = []
        for it in items:
            if not isinstance(it, dict):
                continue
            url = it.get("url") or it.get("link") or ""
            title = it.get("title") or it.get("name") or ""
            content = it.get("content") or it.get("snippet") or it.get("text") or ""
            if url:
                norm.append({"url": url, "title": title, "content": content})
        return norm

# ==============================
# 設定（APIキー）
# ==============================
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()  # すでに先頭で設定済み

# 検索クライアント（環境に応じて自動選択）
search_client = UniversalSearchClient(api_key=OLLAMA_API_KEY)

# ローカル LLM（gemma3:4b）
try:
    # 認証不要のローカル推論用クライアント
    ollama_local_client = ollama.Client()
except Exception as e:
    st.error(f"Ollama ローカルクライアント初期化中にエラー: {e}")
    st.stop()

# ==============================
# 検索（fetchはしない）
# ==============================
@st.cache_data(ttl=1800)
def web_search_and_fetch(query: str, max_results: int = 20) -> List[Dict]:
    """
    fetchは行わず、検索結果（title/url/snippet）だけを返す。
    公式の web_search 結果（title/url/content）を前提に整形。
    """
    results = search_client.search(query, max_results=max_results)

    status = st.status("検索結果を整形中...", expanded=False)
    out: List[Dict] = []
    seen = set()
    for i, r in enumerate(results, 1):
        url = r.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        title = r.get("title") or ""
        content = r.get("content") or ""
        out.append({"title": title, "url": url, "content": content})
        time.sleep(0.01)
    status.update(label=f"✅ {len(out)}件の検索結果を取得", state="complete")
    return out

# ==============================
# ナラティブ生成（gemma3:4b）
# ==============================
def analyze_ai_news_narrative(sources: List[Dict], selected_date: Optional[str]) -> str:
    def clip(txt: str, max_chars=8000):
        return txt[:max_chars] if isinstance(txt, str) else ""

    packed_sources = []
    for i, s in enumerate(sources, 1):
        packed_sources.append(
            f"[ソース{i}]\nタイトル: {s.get('title','')}\nURL: {s.get('url','')}\n内容:\n{clip(s.get('content',''))}\n"
        )

    system = textwrap.dedent("""
    あなたは経験豊富なAI技術ジャーナリストです。
    提供された情報源を深く分析し、読者に分かりやすく魅力的な記事を作成してください。
    
    重要な要件:
    - 各AIニュースについて、独立したセクションを作成する
    - 各セクションには明確な「タイトル」と詳細な「内容」を含める
    - 箇条書きは使用せず、流れるようなナラティブ（物語的）な文章で書く
    - 技術的な内容を一般読者にも理解できるよう噛み砕いて説明する
    - 各ニュースの背景、意義、影響を深く掘り下げる
    - 専門用語には簡単な説明を添える
    - 引用元は文中に自然に織り込む（例: 「〜によると[1]」）
    - 最後に「引用元一覧」セクションを追加し、[番号] URL の形式で列挙
    
    文章スタイル:
    - 読者を引き込む導入文から始める
    - 「である」調の落ち着いた文体
    - 具体例や比喩を用いて理解を促進
    - 各段落は3-5文程度で構成
    - ニュース間の関連性があれば言及する
    - 日本語で出力
    """).strip()

    separator = "\n\n" + "="*50 + "\n\n"
    sources_text = separator.join(packed_sources)
    today = selected_date or datetime.now().strftime("%Y年%m月%d日")

    prompt = f"""
本日（{today}）のAI関連ニュースを以下の情報源から深く分析して日本語で出力してください。

【情報源】
{sources_text}

【出力形式】
# 本日のAI技術ニュース - {today}

[導入部: 今日のAIニュース全体の概要を2-3段落で]

## [ニュース1のタイトル]

[ナラティブ形式の詳細な内容。3-5段落程度]

## [ニュース2のタイトル]

[ナラティブ形式の詳細な内容。3-5段落程度]

[以降、重要なニュースごとに同様のセクションを作成]

---

## 引用元一覧
[1] [URL]
[2] [URL]
...
"""
    full_prompt = f"{system}\n\n{prompt}"

    try:
        response = ollama_local_client.chat(
            model='gemma3:4b',
            messages=[{'role': 'user', 'content': full_prompt}],
            options={'temperature': 0.8, 'num_predict': 8000}
        )
        if isinstance(response, dict):
            content = response.get('message', {}).get('content', '')
        else:
            content = getattr(getattr(response, 'message', None), 'content', '') or ''
        return content or "(応答なし)"
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}\n\n"
        error_msg += "Please ensure Ollama is running locally and gemma3:4b model is installed.\n"
        error_msg += "Run: ollama pull gemma3:4b"
        return error_msg

# ==============================
# UI（入力クエリだけメイン側の“でっかいチャット窓”に移動）
# ==============================
st.markdown('<div class="main-header">AI News Daily</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">最新のAIニュースを検索して、Ollamaで物語調に深掘り解説</div>', unsafe_allow_html=True)

# --- 見出し直下に大きな入力窓を配置（アルゴリズムは変更しない） ---
default_query = f"AI news artificial intelligence latest {datetime.now().strftime('%Y-%m-%d')}"
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

with st.container():
    # 大きく見やすい入力欄（サイドバーではなくメイン）
    st.markdown('<div class="big-chat">', unsafe_allow_html=True)
    main_query = st.text_area(
        label="検索キーワードを入力",
        value=st.session_state.current_query or default_query,
        placeholder="例: latest AI research breakthroughs today",
        height=170,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    # 入力の値だけ更新（実行は従来どおりサイドバーのボタンで行う）
    st.session_state.current_query = (main_query or "").strip()

# サイドバー（クエリ以外はそのまま）
with st.sidebar:
    st.header("検索設定")
    # ※ クエリ入力はここには置かない（場所移動のみ）
    max_results = st.slider("検索件数（上限）", min_value=5, max_value=50, value=20, step=5)
    selected_date = st.date_input("記事の日付（見出し用）", value=datetime.now().date(), format="YYYY/MM/DD")
    selected_date = selected_date.strftime("%Y年%m月%d日") if selected_date else None
    st.markdown("---")
    run_btn = st.button("🔎 ニュース検索 → 🧠 ナラティブ生成", use_container_width=True)

# セッション状態（そのまま）
if 'sources' not in st.session_state: st.session_state.sources = []
if 'article' not in st.session_state: st.session_state.article = ""
if 'current_date' not in st.session_state: st.session_state.current_date = None

# 既存の「実行ボタン」動作は維持
if run_btn:
    st.session_state.article = ""
    st.session_state.sources = []
    st.session_state.current_date = selected_date
    effective_query = st.session_state.current_query or default_query

    with st.spinner("🔍 Web検索を実行中..."):
        sources = web_search_and_fetch(effective_query, max_results=max_results)
        st.session_state.sources = sources

    if sources:
        st.success(f"✅ {len(sources)}件の検索結果を取得しました")
        with st.spinner("🤖 ローカルのgemma3:4bで分析・要約を生成中..."):
            article = analyze_ai_news_narrative(sources, selected_date)
            if article:
                st.session_state.article = article
                st.balloons()
    else:
        st.warning("分析対象となるニュースソースが見つかりませんでした。")

# 出力表示（そのまま）
if st.session_state.article:
    st.markdown("---")
    st.markdown(st.session_state.article, unsafe_allow_html=True)
    with st.expander("参照したニュースソース一覧を見る"):
        for i, s in enumerate(st.session_state.sources, 1):
            st.markdown(f"**{i}. {s.get('title','(no title)')}**")
            st.markdown(f"<{s.get('url','')}>", unsafe_allow_html=True)

st.markdown('<div class="footer">Powered by Ollama Web Search + gemma3:4b</div>', unsafe_allow_html=True)