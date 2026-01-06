from __future__ import annotations

"""Pytest bootstrap for service tests.

Ensures heavy optional dependencies are stubbed so we can import API modules
without installing database drivers/building wheels.
"""

import sys
import types


def _ensure_module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


def _stub_db_modules() -> None:
    # peewee
    if "peewee" not in sys.modules:
        peewee = _ensure_module("peewee")

        class _Exc(Exception):
            pass

        setattr(peewee, "InterfaceError", _Exc)
        setattr(peewee, "OperationalError", _Exc)

        class Field:  # minimal placeholder
            def __init__(self, *args, **kwargs):
                pass

        class TextField(Field):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class IntegerField(Field):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class FloatField(Field):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class DateTimeField(Field):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class BigIntegerField(Field):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class BooleanField(Field):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class CharField(Field):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class Metadata:  # placeholder for peewee.Metadata
            pass

        class CompositeKey:  # placeholder
            def __init__(self, *args, **kwargs):
                pass

        class Model:  # minimal model base
            _meta = types.SimpleNamespace(primary_key=types.SimpleNamespace(field_names=[]))

    for k, v in {
        "Field": Field,
            "TextField": TextField,
            "IntegerField": IntegerField,
            "FloatField": FloatField,
            "DateTimeField": DateTimeField,
            "BigIntegerField": BigIntegerField,
            "BooleanField": BooleanField,
            "CharField": CharField,
            "Metadata": Metadata,
            "CompositeKey": CompositeKey,
        "Model": Model,
    }.items():
        setattr(peewee, k, v)
    # expression helper
    setattr(peewee, "fn", types.SimpleNamespace())
    # additional names used by services
    class Case:  # pragma: no cover
        pass
    setattr(peewee, "Case", Case)
    setattr(peewee, "JOIN", types.SimpleNamespace())

    # playhouse submodules
    if "playhouse.migrate" not in sys.modules:
        playhouse_migrate = _ensure_module("playhouse.migrate")

        class MySQLMigrator:  # placeholder
            def __init__(self, *args, **kwargs):
                pass

        class PostgresqlMigrator:  # placeholder
            def __init__(self, *args, **kwargs):
                pass

        def migrate(*_args, **_kwargs):
            return None

        playhouse_migrate.MySQLMigrator = MySQLMigrator
        playhouse_migrate.PostgresqlMigrator = PostgresqlMigrator
        playhouse_migrate.migrate = migrate

    if "playhouse.pool" not in sys.modules:
        playhouse_pool = _ensure_module("playhouse.pool")

        class _BasePool:  # pragma: no cover
            def __init__(self, *args, **kwargs):
                pass
            def connection_context(self):
                class _Ctx:
                    def __enter__(self):
                        return self
                    def __exit__(self, exc_type, exc, tb):
                        return False
                    def __call__(self, fn):
                        return fn
                return _Ctx()
            def execute_sql(self, *args, **kwargs):
                class _Cursor:
                    def fetchone(self):
                        return (1,)
                return _Cursor()
        class PooledMySQLDatabase(_BasePool):
            pass
        class PooledPostgresqlDatabase(_BasePool):
            pass

        playhouse_pool.PooledMySQLDatabase = PooledMySQLDatabase
        playhouse_pool.PooledPostgresqlDatabase = PooledPostgresqlDatabase


def pytest_sessionstart(session) -> None:  # type: ignore[override]
    _stub_db_modules()
    # Lightweight tiktoken stub to satisfy token_utils import
    if "tiktoken" not in sys.modules:
        tiktoken = types.ModuleType("tiktoken")

        class _Encoder:
            def encode(self, s: str) -> list[int]:
                # naive whitespace tokenization
                return [i for i, _ in enumerate((s or "").split())]

            def decode(self, ids: list[int]) -> str:  # pragma: no cover
                # not used in tests; return placeholder length
                return "".join("x" for _ in ids)

        def get_encoding(_name: str) -> _Encoder:
            return _Encoder()

        tiktoken.get_encoding = get_encoding  # type: ignore[attr-defined]
        sys.modules["tiktoken"] = tiktoken
    # Stub datrie used by rag_tokenizer
    if "datrie" not in sys.modules:
        datrie = types.ModuleType("datrie")

        class Trie(dict):  # pragma: no cover - not used in tests
            def __init__(self, *args, **kwargs):
                super().__init__()
            def __setitem__(self, key, value):
                super().__setitem__(str(key), value)
            def __getitem__(self, key):
                return super().__getitem__(str(key))

        datrie.Trie = Trie  # type: ignore[attr-defined]
        sys.modules["datrie"] = datrie
    # Stub hanziconv used by rag_tokenizer
    if "hanziconv" not in sys.modules:
        hanziconv = types.ModuleType("hanziconv")
        class HanziConv:
            @staticmethod
            def toSimplified(s: str) -> str:  # pragma: no cover
                return s
            @staticmethod
            def toTraditional(s: str) -> str:  # pragma: no cover
                return s
        hanziconv.HanziConv = HanziConv  # type: ignore[attr-defined]
        sys.modules["hanziconv"] = hanziconv
    # Stub nltk tokenizer
    if "nltk" not in sys.modules:
        nltk = types.ModuleType("nltk")
        def word_tokenize(s: str) -> list[str]:  # pragma: no cover
            return (s or "").split()
        nltk.word_tokenize = word_tokenize  # type: ignore[attr-defined]
        # stem submodule
        stem = types.ModuleType("nltk.stem")
        class PorterStemmer:  # pragma: no cover
            def stem(self, s: str) -> str:
                return s
        class WordNetLemmatizer:  # pragma: no cover
            def lemmatize(self, s: str) -> str:
                return s
        stem.PorterStemmer = PorterStemmer  # type: ignore[attr-defined]
        stem.WordNetLemmatizer = WordNetLemmatizer  # type: ignore[attr-defined]
        sys.modules["nltk"] = nltk
        sys.modules["nltk.stem"] = stem
    # Stub ragflow.rag.nlp package to avoid heavy deps
    for pkg_name in ("rag.nlp", "ragflow.rag.nlp"):
        if pkg_name in sys.modules:
            continue
        nlp = types.ModuleType(pkg_name)
        # is_english: simple ASCII check
        def is_english(texts):  # pragma: no cover
            if isinstance(texts, str):
                texts = list(texts)
            if not texts:
                return False
            import re as _re
            pat = _re.compile(r"[A-Za-z0-9\s.,']+")
            return all(pat.fullmatch(t.strip()) for t in texts if isinstance(t, str))
        def is_chinese(text: str):  # pragma: no cover
            if not text:
                return False
            return any('\u4e00' <= ch <= '\u9fff' for ch in text)
        def extract_between(s: str, start: str, end: str) -> str:  # pragma: no cover
            try:
                i = s.index(start) + len(start)
                j = s.index(end, i)
                return s[i:j]
            except Exception:
                return ""
        class _Tokenizer:  # pragma: no cover
            @staticmethod
            def tokenize(s: str) -> list[str]:
                return (s or "").split()
            @staticmethod
            def fine_grained_tokenize(tokens: list[str]) -> list[str]:
                return tokens
        class _Search:  # pragma: no cover
            @staticmethod
            def index_name(tenant_id: str) -> str:
                return f"idx-{tenant_id}"
        nlp.is_english = is_english  # type: ignore[attr-defined]
        nlp.is_chinese = is_chinese  # type: ignore[attr-defined]
        nlp.rag_tokenizer = _Tokenizer()  # type: ignore[attr-defined]
        nlp.search = _Search()  # type: ignore[attr-defined]
        nlp.extract_between = extract_between  # type: ignore[attr-defined]
        sys.modules[pkg_name] = nlp
    # Stub doc store connectors to avoid network/driver imports
    for mod_name, cls_name in (
        ("rag.utils.infinity_conn", "InfinityConnection"),
        ("rag.utils.es_conn", "ESConnection"),
        ("rag.utils.opensearch_conn", "OSConnection"),
        ("rag.utils.redis_conn", None),
        ("rag.utils.storage_factory", None),
    ):
        if mod_name in sys.modules:
            continue
        m = types.ModuleType(mod_name)
        if cls_name:
            class _Conn:  # pragma: no cover
                def __init__(self, *args, **kwargs):
                    pass
            setattr(m, cls_name, _Conn)
        else:
            # Provide REDIS_CONN placeholder
            class _Redis:  # pragma: no cover
                pass
            m.REDIS_CONN = _Redis()  # type: ignore[attr-defined]
            # Provide STORAGE_IMPL placeholder
            m.STORAGE_IMPL = object()  # type: ignore[attr-defined]
        sys.modules[mod_name] = m
    # Stub trio (not used in our focused tests)
    if "trio" not in sys.modules:
        sys.modules["trio"] = types.ModuleType("trio")
    # Stub litellm
    if "litellm" not in sys.modules:
        sys.modules["litellm"] = types.ModuleType("litellm")
    # Stub openai
    if "openai" not in sys.modules:
        openai = types.ModuleType("openai"); openai.__path__ = []  # type: ignore[attr-defined]
        class OpenAI:  # pragma: no cover
            pass
        openai.OpenAI = OpenAI  # type: ignore[attr-defined]
        sys.modules["openai"] = openai
        # openai.lib.azure.AzureOpenAI
        openai_lib = types.ModuleType("openai.lib"); openai_lib.__path__ = []  # type: ignore[attr-defined]
        openai_azure = types.ModuleType("openai.lib.azure")
        class AzureOpenAI:  # pragma: no cover
            pass
        openai_azure.AzureOpenAI = AzureOpenAI  # type: ignore[attr-defined]
        sys.modules["openai.lib"] = openai_lib
        sys.modules["openai.lib.azure"] = openai_azure
    # Stub zhipuai
    if "zhipuai" not in sys.modules:
        zhipuai = types.ModuleType("zhipuai")
        class ZhipuAI:  # pragma: no cover
            pass
        zhipuai.ZhipuAI = ZhipuAI  # type: ignore[attr-defined]
        sys.modules["zhipuai"] = zhipuai
    # Stub dashscope
    if "dashscope" not in sys.modules:
        sys.modules["dashscope"] = types.ModuleType("dashscope")
    # Stub google.generativeai
    if "google.generativeai" not in sys.modules:
        google = types.ModuleType("google"); google.__path__ = []  # type: ignore[attr-defined]
        genai = types.ModuleType("google.generativeai")
        sys.modules["google"] = google
        sys.modules["google.generativeai"] = genai
    # Stub ollama
    if "ollama" not in sys.modules:
        ollama = types.ModuleType("ollama")
        class Client:  # pragma: no cover
            pass
        ollama.Client = Client  # type: ignore[attr-defined]
        sys.modules["ollama"] = ollama
    # Stub pluginlib
    if "pluginlib" not in sys.modules:
        pluginlib = types.ModuleType("pluginlib")
        def Parent(*_args, **_kwargs):  # pragma: no cover
            def deco(fn):
                return fn
            return deco
        pluginlib.Parent = Parent  # type: ignore[attr-defined]
        pluginlib.abstractmethod = Parent  # type: ignore[attr-defined]
        sys.modules["pluginlib"] = pluginlib
    # Stub langfuse
    if "langfuse" not in sys.modules:
        langfuse = types.ModuleType("langfuse")
        class Langfuse:  # pragma: no cover
            pass
        langfuse.Langfuse = Langfuse  # type: ignore[attr-defined]
        sys.modules["langfuse"] = langfuse
    # Stub tavily
    if "tavily" not in sys.modules:
        tavily = types.ModuleType("tavily")
        class TavilyClient:  # pragma: no cover
            def __init__(self, *args, **kwargs):
                pass
            def search(self, *args, **kwargs):
                return {"results": []}
        tavily.TavilyClient = TavilyClient  # type: ignore[attr-defined]
        sys.modules["tavily"] = tavily
    # Stub mcp client sessions
    if "mcp.client.session" not in sys.modules:
        mcp = types.ModuleType("mcp"); mcp.__path__ = []  # type: ignore[attr-defined]
        client = types.ModuleType("mcp.client"); client.__path__ = []  # type: ignore[attr-defined]
        session = types.ModuleType("mcp.client.session")
        class ClientSession:  # pragma: no cover
            pass
        session.ClientSession = ClientSession  # type: ignore[attr-defined]
        sys.modules["mcp"] = mcp
        sys.modules["mcp.client"] = client
        sys.modules["mcp.client.session"] = session
        sse = types.ModuleType("mcp.client.sse")
        def sse_client(*_args, **_kwargs):
            return None
        sse.sse_client = sse_client  # type: ignore[attr-defined]
        sys.modules["mcp.client.sse"] = sse
        stream = types.ModuleType("mcp.client.streamable_http")
        def streamablehttp_client(*_args, **_kwargs):
            return None
        stream.streamablehttp_client = streamablehttp_client  # type: ignore[attr-defined]
        sys.modules["mcp.client.streamable_http"] = stream
        mcp_types = types.ModuleType("mcp.types")
        class CallToolResult:  # pragma: no cover
            pass
        class ListToolsResult:  # pragma: no cover
            pass
        class TextContent:  # pragma: no cover
            pass
        class Tool:  # pragma: no cover
            pass
        mcp_types.CallToolResult = CallToolResult  # type: ignore[attr-defined]
        mcp_types.ListToolsResult = ListToolsResult  # type: ignore[attr-defined]
        mcp_types.TextContent = TextContent  # type: ignore[attr-defined]
        mcp_types.Tool = Tool  # type: ignore[attr-defined]
        sys.modules["mcp.types"] = mcp_types
    # Preload chunk_app directly to avoid importing the entire apps package graph
    try:
        import importlib.util as _ilu
        from pathlib import Path as _Path
        from flask import Flask as _Flask, Blueprint as _Blueprint
        root = _Path(__file__).resolve().parents[2]
        chunk_path = root / "ragflow" / "api" / "apps" / "chunk_app.py"
        if chunk_path.exists() and "ragflow.api.apps.chunk_app" not in sys.modules:
            spec = _ilu.spec_from_file_location("ragflow.api.apps.chunk_app", str(chunk_path))
            if spec and spec.loader:
                mod = _ilu.module_from_spec(spec)
                # Inject minimal globals expected by app modules
                setattr(mod, "app", _Flask("ragflow-test"))
                setattr(mod, "manager", _Blueprint("chunk", "ragflow.api.apps.chunk_app"))
                sys.modules["ragflow.api.apps.chunk_app"] = mod
                spec.loader.exec_module(mod)
                # Provide a lightweight package module exposing chunk_app
                pkg_name = "ragflow.api.apps"
                if pkg_name not in sys.modules:
                    pkg = types.ModuleType(pkg_name); pkg.__path__ = []  # type: ignore[attr-defined]
                    sys.modules[pkg_name] = pkg
                setattr(sys.modules[pkg_name], "chunk_app", mod)
    except Exception:
        # Fallback to normal import path if preload fails
        pass
