# ai_news_dashboard.py
import streamlit as st
import os
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
# 検索クライアント（fetchせず、検索結果だけ返す）
# 1) ollama.Client.web_search が使えれば最優先
# 2) OLLAMA_WEB_SEARCH_URL / OLLAMA_WEB_BASE_URL 経由のHTTP（失敗しても例外にしない）
# 3) 無鍵フォールバック：Google News RSS / Bing News RSS（bs4/docduck不使用）
# 4) さらに鍵があれば Serper / Brave / Bing API も利用可能（任意）
# ==============================
class UniversalSearchClient:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        self.api_key = (api_key or os.getenv("OLLAMA_API_KEY", "")).strip()
        self.timeout = timeout

        # ---- Ollama SDK（web_search があれば使う）
        self._sdk_client = None
        self._has_sdk_search = False
        try:
            sdk_host = os.getenv("OLLAMA_HOST", "") or os.getenv("OLLAMA_WEB_BASE_URL", "")
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
            self._sdk_client = ollama.Client(host=sdk_host.strip() or None, headers=headers)
            self._has_sdk_search = hasattr(self._sdk_client, "web_search")
        except Exception:
            self._sdk_client = None
            self._has_sdk_search = False

        # ---- HTTP直叩き用（Ollama系）
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Ollama-Api-Key": self.api_key,
                "X-API-Key": self.api_key,
            })
        self.session.headers.update({
            "Accept": "application/json,text/plain,*/*",
            "User-Agent": "Mozilla/5.0 (compatible; AI-News-Dashboard/1.0)",
        })

        self.fixed_search_url = (os.getenv("OLLAMA_WEB_SEARCH_URL", "")).strip() or None
        base_env = (os.getenv("OLLAMA_WEB_BASE_URL", "") or os.getenv("OLLAMA_HOST", "")).strip()
        self.base_candidates: List[str] = []
        if base_env:
            self.base_candidates.append(base_env.rstrip("/"))
        self.base_candidates += [
            "https://api.ollama.com",
            "https://api.ollama.ai",
            "https://cloud.ollama.com",
            "http://localhost:11434",
        ]
        self.search_paths = [
            "/web/search", "/api/web/search",
            "/web_search", "/api/web_search",
            "/search", "/api/search",
            "/v1/web/search", "/api/v1/web/search",
            "/v1/search", "/api/v1/search",
        ]

        # ---- 代替の外部検索API（任意）
        self.serper_key = os.getenv("SERPER_API_KEY", "").strip()
        self.brave_key  = os.getenv("BRAVE_API_KEY", "").strip()
        self.bing_key   = os.getenv("BING_SUBSCRIPTION_KEY", "").strip()

    # -------- 1) Ollama SDK --------
    def _try_ollama_sdk(self, query: str, max_results: int) -> Optional[List[Dict]]:
        if not (self._sdk_client and self._has_sdk_search):
            return None
        try:
            res = self._sdk_client.web_search(query, max_results=max_results)
            # normalize
            if hasattr(res, "results"):
                out = []
                for it in res.results:
                    url = getattr(it, "url", "") or ""
                    title = getattr(it, "title", "") or ""
                    content = getattr(it, "content", "") or ""
                    if url:
                        out.append({"url": url, "title": title, "content": content})
                return out
            elif isinstance(res, dict) or isinstance(res, list):
                return self._normalize_search(res)
        except Exception:
            return None
        return None

    # -------- 2) Ollama HTTP（失敗しても例外は投げない） --------
    def _try_ollama_http(self, query: str, max_results: int) -> Optional[List[Dict]]:
        # 固定URL
        if self.fixed_search_url:
            r = self._http_search_once(self.fixed_search_url, query, max_results)
            if r:
                return r
        # base x path
        for b in self.base_candidates:
            for p in self.search_paths:
                url = f"{b.rstrip('/')}{p}"
                r = self._http_search_once(url, query, max_results)
                if r:
                    return r
        return None

    def _http_search_once(self, url: str, query: str, max_results: int) -> Optional[List[Dict]]:
        # GET
        for params in (
            {"q": query, "k": max_results},
            {"query": query, "k": max_results},
            {"q": query, "limit": max_results},
            {"query": query, "max_results": max_results},
        ):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception:
                        data = {}
                    items = self._normalize_search(data)
                    if items:
                        return items
            except Exception:
                pass
        # POST
        for payload in (
            {"q": query, "k": max_results},
            {"query": query, "k": max_results},
            {"q": query, "limit": max_results},
            {"query": query, "max_results": max_results},
            {"q": query, "limit": max_results, "max_results": max_results},
        ):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception:
                        data = {}
                    items = self._normalize_search(data)
                    if items:
                        return items
            except Exception:
                pass
        return None

    # -------- 3) 無鍵RSS（確実に結果を返す） --------
    def _try_google_news_rss(self, query: str, max_results: int) -> Optional[List[Dict]]:
        q = f"{query} when:7d"
        q_enc = urllib.parse.quote_plus(q)
        url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-US&gl=US&ceid=US:en"
        try:
            r = requests.get(url, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200 or not r.text:
                return None
            return self._parse_rss_items(r.text, max_results)
        except Exception:
            return None

    def _try_bing_news_rss(self, query: str, max_results: int) -> Optional[List[Dict]]:
        q_enc = urllib.parse.quote_plus(query)
        url = f"https://www.bing.com/news/search?q={q_enc}&format=rss"
        try:
            r = requests.get(url, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200 or not r.text:
                return None
            return self._parse_rss_items(r.text, max_results)
        except Exception:
            return None

    @staticmethod
    def _parse_rss_items(xml_text: str, max_results: int) -> List[Dict]:
        items: List[Dict] = []
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return items
        for it in root.findall(".//item"):
            title = html.unescape(it.findtext("title") or "")
            link = (it.findtext("link") or "").strip()
            desc = html.unescape(it.findtext("description") or "")
            if link:
                items.append({"title": title, "url": link, "content": desc})
            if len(items) >= max_results:
                break
        return items

    # -------- 4) 追加の外部API（任意） --------
    def _try_serper(self, query: str, max_results: int) -> Optional[List[Dict]]:
        if not self.serper_key:
            return None
        try:
            url = "https://google.serper.dev/search"
            headers = {"X-API-KEY": self.serper_key, "Content-Type": "application/json"}
            payload = {"q": query, "num": max_results}
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                return None
            data = r.json()
            items = []
            for it in data.get("organic", [])[:max_results]:
                items.append({
                    "title": it.get("title") or "",
                    "url": it.get("link") or "",
                    "content": it.get("snippet") or "",
                })
            if len(items) < max_results:
                news_url = "https://google.serper.dev/news"
                r2 = requests.post(news_url, headers=headers, json=payload, timeout=self.timeout)
                if r2.status_code == 200:
                    d2 = r2.json()
                    for it in d2.get("news", []):
                        if len(items) >= max_results:
                            break
                        items.append({
                            "title": it.get("title") or "",
                            "url": it.get("link") or "",
                            "content": it.get("snippet") or "",
                        })
            return [x for x in items if x["url"]]
        except Exception:
            return None

    def _try_brave(self, query: str, max_results: int) -> Optional[List[Dict]]:
        if not self.brave_key:
            return None
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {"X-Subscription-Token": self.brave_key, "Accept": "application/json"}
            params = {"q": query, "count": max_results}
            r = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            if r.status_code != 200:
                return None
            data = r.json()
            items = []
            for it in data.get("web", {}).get("results", [])[:max_results]:
                items.append({
                    "title": it.get("title") or "",
                    "url": it.get("url") or "",
                    "content": it.get("description") or "",
                })
            return [x for x in items if x["url"]]
        except Exception:
            return None

    def _try_bing(self, query: str, max_results: int) -> Optional[List[Dict]]:
        if not self.bing_key:
            return None
        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.bing_key}
            params = {"q": query, "count": max_results, "responseFilter": "Webpages"}
            r = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            if r.status_code != 200:
                return None
            data = r.json()
            items = []
            for it in data.get("webPages", {}).get("value", [])[:max_results]:
                items.append({
                    "title": it.get("name") or "",
                    "url": it.get("url") or "",
                    "content": it.get("snippet") or "",
                })
            return [x for x in items if x["url"]]
        except Exception:
            return None

    # -------- 公開：検索 --------
    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        # 1) Ollama SDK
        res = self._try_ollama_sdk(query, max_results)
        if res:
            return res[:max_results]

        # 2) Ollama HTTP
        res = self._try_ollama_http(query, max_results)
        if res:
            return res[:max_results]

        # # 3) 無鍵RSS（まずGoogle News、ダメならBing News）
        # res = self._try_google_news_rss(query, max_results)
        # if res:
        #     return res[:max_results]
        # res = self._try_bing_news_rss(query, max_results)
        # if res:
        #     return res[:max_results]

        # # 4) 任意の鍵があれば外部API
        # for fn in (self._try_serper, self._try_brave, self._try_bing):
        #     res = fn(query, max_results)
        #     if res:
        #         return res[:max_results]

        # すべて失敗
        return []

    # -------- 正規化 --------
    @staticmethod
    def _normalize_search(data) -> List[Dict]:
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if isinstance(data.get("results"), list):
                items = data["results"]
            elif isinstance(data.get("data"), list):
                items = data["data"]
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
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip() or "ここにOllama apiを載せてください"
os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY

# 検索クライアント（環境に応じて自動選択）
search_client = UniversalSearchClient(api_key=OLLAMA_API_KEY)

# ローカル LLM（gemma3:4b）
try:
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
    RSSフォールバックにより、鍵なしでも>0件を返すことを目標に実装。
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
st.markdown('<div class="subtitle">最新のAIニュースを検索して、gemma3:4bで物語調に深掘り解説</div>', unsafe_allow_html=True)

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