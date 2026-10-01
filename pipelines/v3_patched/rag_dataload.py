import json
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import jieba
import csv
from collections import Counter
from typing import List, Dict, Tuple, Optional

# ==========================================
# NLTK RESOURCE BOOTSTRAP
# FIX 1: Original checked all resources under 'corpora/', but `punkt` lives at
# 'tokenizers/punkt', not 'corpora/punkt'. This caused a LookupError on every
# import, triggering an unnecessary re-download attempt each run. Newer NLTK
# also requires 'punkt_tab'. Each resource now uses its correct path prefix.
#
# FIX 15b: This bootstrap must NEVER crash the import. Two failure modes seen
# in the wild:
#   (a) LookupError — resource genuinely missing (handled by re-download).
#   (b) OSError: '.../punkt/PY3_tab' — an NLTK version-layout bug (see
#       nltk/nltk#3305) where the loader looks for a _tab-suffixed path INSIDE
#       the legacy punkt/ folder instead of the sibling punkt_tab/ directory,
#       OR downloaded .zip archives were never auto-extracted. `nltk.data.find`
#       raises OSError (NOT LookupError) in this case, so the original
#       except-LookupError bootstrap did not catch it and the import crashed.
# We now (1) proactively extract any stale *.zip archives sitting in the
# nltk_data tokenizers/corpora dirs, and (2) catch BOTH exception types and
# degrade gracefully. English tokenization has a punkt-free regex fallback in
# clean_and_tokenize() (fix #15), so the module imports even if punkt cannot
# be resolved; English results then differ from the paper (FIXES.md #30).
# ==========================================
import os as _os
import zipfile as _zipfile

def _extract_stale_nltk_zips():
    """Auto-extract any *.zip in nltk_data subdirs that were downloaded but
    not unpacked (a known cause of the punkt/PY3_tab OSError)."""
    for _root in nltk.data.path:
        for _sub in ('tokenizers', 'corpora'):
            _d = _os.path.join(_root, _sub)
            if not _os.path.isdir(_d):
                continue
            for _fn in _os.listdir(_d):
                if _fn.endswith('.zip'):
                    _stem = _fn[:-4]
                    _target = _os.path.join(_d, _stem)
                    if not _os.path.isdir(_target):
                        try:
                            _zipfile.ZipFile(_os.path.join(_d, _fn)).extractall(_d)
                        except Exception:
                            pass  # best-effort; fallback handles the rest

try:
    _extract_stale_nltk_zips()
except Exception:
    pass

_NLTK_RESOURCES = [
    ('corpora/stopwords',    'stopwords'),
    ('tokenizers/punkt',     'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),   # required in NLTK >= 3.9
    ('corpora/wordnet',      'wordnet'),
]
for _path, _name in _NLTK_RESOURCES:
    try:
        nltk.data.find(_path)
    except (LookupError, OSError):
        # LookupError: missing. OSError: present-but-broken layout (PY3_tab).
        try:
            nltk.download(_name, quiet=True)
            # re-extract in case the fresh download also arrived zipped
            _extract_stale_nltk_zips()
        except Exception:
            pass  # never let resource bootstrap crash import — see FIX 15b


# ==========================================
# COLUMN DEFINITIONS
# ==========================================
COL_NAMES = [
    'TopNumber', 'AirlineName', 'ReviewerName', 'Rating', 'ReviewDate', 'ReviewTitle',
    'ReviewText', 'Tags', 'DateofTravel', 'Aspects', 'ResponserName', 'ResponseDate',
    'ResponseText', 'ReviewerProfileUrl', 'UserReviewLink', 'AirlineReviewLink', 'CrawlTime'
]

LANGUAGES_TO_TRAIN = {
    'English': ('Translated_ReviewTitle_English', 'Translated_ReviewText_English'),
    # Note: 'Simplied' (sic) matches the actual column name in the source CSV.
    'Simplified_Chinese': ('ReviewTitle_Simplied_Chinese', 'ReviewText_Simplied_Chinese'),
}

# ==========================================
# FIX 2: COMPREHENSIVE CHINESE STOPWORD LIST
# The original had only 10 entries. High-frequency function words not in that
# list (在, 中, 不, 这, 那, 有, 可以, 因为, ...) dominated Gibbs sampling and
# produced incoherent topics — the very failure mode the paper claims to solve.
# This list covers particles, pronouns, conjunctions, prepositions, auxiliary
# verbs, discourse markers, and temporal/spatial words (~150 entries).
# For production use, supplement with a domain-specific file via:
#   loader = UniversalRAGLoader(extra_zh_stopwords=load_stopwords("aviation_stop.txt"))
# ==========================================
_ZH_STOPWORDS_BASE = {
    # Particles / structural
    '的', '了', '着', '过', '地', '得', '呢', '吧', '嘛', '啊', '哦', '啦', '哈',
    '吗', '嗯', '喔', '哎', '哇',
    # Pronouns
    '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们',
    '自己', '自身', '本身', '这', '那', '这个', '那个', '这些', '那些', '此',
    # Conjunctions
    '和', '与', '及', '而', '或', '但', '而且', '但是', '然而', '虽然', '尽管',
    '即使', '不管', '无论', '因为', '所以', '因此', '于是', '从而', '如果',
    '假如', '除非', '只要', '只有', '既然', '并且', '或者', '还是', '不但',
    '不仅', '另外', '此外', '同时', '首先', '其次', '最后', '总之', '总体',
    '然后', '接着', '以及',
    # Prepositions
    '在', '从', '向', '到', '对', '于', '为', '以', '被', '把', '让', '给',
    '跟', '同', '比', '除', '关于', '对于', '由于', '按照', '根据', '通过',
    '进行', '关于', '对于',
    # Auxiliary verbs / adverbs
    '是', '有', '不', '也', '就', '都', '将', '使', '被', '能', '会', '可以',
    '应该', '需要', '应当', '必须', '能够', '可能', '已经', '正在', '将要',
    '曾经', '刚刚', '一直', '总是', '还', '又', '再', '已', '很', '更', '最',
    '太', '非常', '比较', '真的', '确实', '大概', '也许', '几乎', '完全',
    # Determiners / quantifiers
    '一', '各', '每', '某', '其', '该', '此', '之', '些', '等', '等等',
    '一些', '一个', '一种', '这样', '那样', '如此',
    # Spatial / temporal
    '上', '下', '左', '右', '前', '后', '里', '外', '内', '间', '中',
    '年', '月', '日', '时', '分', '秒', '第', '个', '件', '次', '种',
    # Question words
    '什么', '怎么', '为什么', '哪', '哪里', '谁', '哪些', '多少', '几',
    # Filler / discourse
    '当', '其中', '其他', '另', '其实', '事实上',
}

# ==========================================
# JAPANESE STOPWORD LIST
# Covers particles (は、が、を、に、で、と、も、の、へ、から、まで、より),
# auxiliary verbs (です、ます、ない、た、て、いる、ある、する、なる),
# pronouns, conjunctions, and common discourse markers.
# Japanese tokenization via MeCab (fugashi) segments at morpheme level,
# so single-character particles need explicit filtering.
# ==========================================
_JA_STOPWORDS_BASE = {
    # Particles (助詞)
    'は', 'が', 'を', 'に', 'で', 'と', 'も', 'の', 'へ', 'から', 'まで',
    'より', 'や', 'か', 'な', 'ね', 'よ', 'わ', 'ぞ', 'ぜ', 'さ', 'こそ',
    'だけ', 'しか', 'ほど', 'くらい', 'ぐらい', 'など', 'なんか', 'なんて',
    'って', 'という', 'とか', 'けど', 'けれど', 'けれども', 'ので', 'のに',
    'ながら', 'ために', 'として', 'について', 'において', 'に関して',
    'に対して', 'によって', 'によると', 'に従って',
    # Auxiliary verbs / copula (助動詞・補助動詞)
    'です', 'ます', 'ない', 'ません', 'でした', 'ました', 'た', 'て', 'で',
    'いる', 'ある', 'する', 'なる', 'れる', 'られる', 'せる', 'させる',
    'しまう', 'しまい', 'いく', 'くる', 'おく', 'みる', 'もらう', 'あげる',
    'くれる', 'やる', 'いただく', 'ください', 'なさい', 'だ', 'である',
    # Pronouns (代名詞)
    'これ', 'それ', 'あれ', 'どれ', 'ここ', 'そこ', 'あそこ', 'どこ',
    'この', 'その', 'あの', 'どの', 'こんな', 'そんな', 'あんな', 'どんな',
    'わたし', 'わたしたち', 'ぼく', 'おれ', 'あなた', 'きみ', 'かれ', 'かのじょ',
    'かれら', 'じぶん', 'みんな', 'みなさん',
    # Conjunctions (接続詞)
    'そして', 'また', 'および', 'または', 'あるいは', 'しかし', 'でも', 'ただし',
    'けれども', 'だが', 'ところが', 'それでも', 'なのに', 'それに', 'その上',
    'さらに', 'そのため', 'したがって', 'だから', 'なぜなら', 'もし', 'もしも',
    'たとえ', 'かりに', 'なお', 'ちなみに', 'つまり', 'すなわち', 'たとえば',
    'まず', '次に', 'それから', '最後に', 'いっぽう', '一方',
    # Adverbs / degree words (副詞)
    'とても', 'すごく', 'かなり', 'かなり', 'もっと', 'さらに', 'より',
    'あまり', 'そんなに', 'こんなに', 'どんなに', 'ほとんど', 'だいたい',
    'おおよそ', 'ちょっと', 'すこし', '少し', 'やや', 'はるかに',
    'まだ', 'もう', 'すでに', 'やっと', 'ようやく', 'ついに', 'とうとう',
    'いつも', 'いつか', 'ときどき', 'たまに', 'けっして', 'ぜったいに',
    # Numbers / counters commonly used as noise
    'こと', 'もの', 'ため', 'よう', 'ところ', 'わけ', 'はず', 'つもり',
    'ほう', 'たち', 'など', 'ほか', 'うち', 'なか', 'まま', 'ばかり',
}

# ==========================================
# THAI STOPWORD BASE LIST
# Used as a fallback when pythainlp is not installed, and importable by
# rag_preprocessing.py so both classes share the same stopword source.
# pythainlp's built-in thai_stopwords() is more comprehensive (~800 entries)
# and is used when available; this list covers the most critical function words.
# ==========================================
_TH_STOPWORDS_BASE = {
    # Particles / markers
    'ที่', 'และ', 'ใน', 'ของ', 'มี', 'ได้', 'จะ', 'ว่า', 'แต่', 'หรือ',
    'ให้', 'การ', 'กับ', 'จาก', 'เป็น', 'ไม่', 'นี้', 'นั้น', 'ก็', 'แล้ว',
    'ซึ่ง', 'โดย', 'เพื่อ', 'เมื่อ', 'ถ้า', 'อยู่', 'ออก', 'มา', 'ไป', 'ยัง',
    'อีก', 'ต้อง', 'อย่าง', 'ทั้ง', 'ทำ', 'ไม่ได้', 'เพราะ', 'แม้', 'หาก',
    # Pronouns
    'ผม', 'ฉัน', 'เรา', 'คุณ', 'เขา', 'เธอ', 'มัน', 'พวกเขา', 'ตัวเอง',
    'ใคร', 'อะไร', 'ที่ไหน', 'อย่างไร', 'เมื่อไร', 'ทำไม',
    # Common function words
    'แห่ง', 'ด้วย', 'ตาม', 'ระหว่าง', 'ใต้', 'บน', 'ล่าง', 'หน้า', 'หลัง',
    'ขึ้น', 'ลง', 'เข้า', 'นอก', 'ใกล้', 'ไกล', 'ทุก', 'บาง', 'หลาย',
    'มาก', 'น้อย', 'อีก', 'แรก', 'สุด', 'ต่อ', 'ก่อน', 'หลัง', 'ระหว่าง',
}


class UniversalRAGLoader:
    """
    Universal data loader, tokeniser, chunker, and BoW builder for Lex-TM.

    Supported languages: English ('en'), Chinese ('zh'), Japanese ('ja'),
    Thai ('th').

    Parameters
    ----------
    chunk_size : int
        Number of tokens per chunk for English. Default 150.
    chunk_size_zh : int or None
        Tokens per Chinese chunk. Defaults to chunk_size * 2.
    chunk_size_ja : int or None
        Tokens per Japanese chunk (MeCab morphemes). Defaults to chunk_size * 2.
        MeCab morphemes are short (avg ~1-2 chars), so doubling maintains
        comparable semantic density to English chunks.
    chunk_size_th : int or None
        Tokens per Thai chunk (PyThaiNLP tokens). Defaults to chunk_size * 2.
    overlap : int
        Sliding-window overlap in tokens (same unit as the relevant chunk_size).
    min_df : int
        Minimum chunk-frequency for vocabulary inclusion. Default 2.
    extra_zh_stopwords : set or None
        Additional Chinese stopwords merged at init time.
    extra_en_stopwords : set or None
        Additional English stopwords merged at init time.
    extra_ja_stopwords : set or None
        Additional Japanese stopwords merged at init time.
    extra_th_stopwords : set or None
        Additional Thai stopwords merged at init time.

    Japanese dependency
    -------------------
        pip install fugashi ipadic
    MeCab requires no system binary when installed via the ipadic Python package.

    Thai dependency
    ---------------
        pip install pythainlp
    PyThaiNLP is pure Python with no system dependencies.
    """

    def __init__(
        self,
        chunk_size: int = 150,
        chunk_size_zh: Optional[int] = None,
        chunk_size_ja: Optional[int] = None,
        chunk_size_th: Optional[int] = None,
        overlap: int = 30,
        min_df: int = 2,
        extra_zh_stopwords: Optional[set] = None,
        extra_en_stopwords: Optional[set] = None,
        extra_ja_stopwords: Optional[set] = None,
        extra_th_stopwords: Optional[set] = None,
    ):
        self.chunk_size    = chunk_size
        self.chunk_size_zh = chunk_size_zh if chunk_size_zh is not None else chunk_size * 2
        self.chunk_size_ja = chunk_size_ja if chunk_size_ja is not None else chunk_size * 2
        self.chunk_size_th = chunk_size_th if chunk_size_th is not None else chunk_size * 2
        self.overlap = overlap
        self.min_df  = min_df

        self.lemmatizer = WordNetLemmatizer()

        # English stopwords
        self.en_stopwords = set(stopwords.words('english'))
        if extra_en_stopwords:
            self.en_stopwords.update(extra_en_stopwords)

        # Chinese stopwords
        self.zh_stopwords = set(_ZH_STOPWORDS_BASE)
        if extra_zh_stopwords:
            self.zh_stopwords.update(extra_zh_stopwords)

        # Japanese stopwords
        self.ja_stopwords = set(_JA_STOPWORDS_BASE)
        if extra_ja_stopwords:
            self.ja_stopwords.update(extra_ja_stopwords)

        # Japanese tokenizer — lazy import so fugashi is only required when
        # actually processing Japanese text.
        try:
            import fugashi
            #self._ja_tagger = fugashi.Tagger()
            import ipadic
            # Pass ipadic.MECAB_ARGS to force fugashi to use the local Python 
            # dictionary and bypass the hardcoded Windows C:\ check.
            self._ja_tagger = fugashi.GenericTagger(ipadic.MECAB_ARGS)
        except ImportError:
            self._ja_tagger = None

        # Thai tokenizer and stopwords — lazy import.
        # pythainlp's thai_stopwords() (~800 entries) is used when available.
        # _TH_STOPWORDS_BASE is the fallback when pythainlp is not installed,
        # and is also importable by rag_preprocessing.py for consistency.
        try:
            from pythainlp.tokenize import word_tokenize as _th_tok
            from pythainlp.corpus.common import thai_stopwords as _th_sw
            self._th_tokenize = _th_tok
            self.th_stopwords = set(_th_sw())
        except ImportError:
            self._th_tokenize = None
            self.th_stopwords = set(_TH_STOPWORDS_BASE)   # fallback

        if extra_th_stopwords:
            self.th_stopwords.update(extra_th_stopwords)

        # Vocabulary state (populated by process_corpus)
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.vocab_size: int = 0

    # ==========================================
    # 1. DATASET-SPECIFIC LOADERS
    # ==========================================

    def load_english_aviation(self, filepath: str) -> pd.DataFrame:
        """Loads the headerless TSV aviation review file."""
        print(f"Loading English Aviation from: {filepath}")
        try:
            df = pd.read_csv(
                filepath, sep='\t', header=None, names=COL_NAMES,
                quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip'
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read English Aviation file: {e}") from e

        df['document_id'] = 'en_aviation_' + df.index.astype(str)
        df['ReviewTitle'] = df['ReviewTitle'].fillna('')
        df['ReviewText']  = df['ReviewText'].fillna('')
        df['full_text']   = df['ReviewTitle'] + '. ' + df['ReviewText']
        return df[['document_id', 'ReviewTitle', 'full_text']].dropna(subset=['full_text'])

    def load_chinese_aviation(self, filepath: str) -> pd.DataFrame:
        """Loads the multilingual aviation CSV and extracts Simplified Chinese columns."""
        print(f"Loading Chinese Aviation from: {filepath}")
        df = pd.read_csv(filepath, header=0, on_bad_lines='warn')

        title_col, text_col = LANGUAGES_TO_TRAIN['Simplified_Chinese']

        # FIX 5: Validate that expected columns exist before proceeding
        missing = {title_col, text_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"Chinese Aviation CSV is missing columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        df['document_id'] = 'zh_aviation_' + df.index.astype(str)
        df[title_col]     = df[title_col].fillna('')
        df[text_col]      = df[text_col].fillna('')
        df['ReviewTitle'] = df[title_col]
        df['full_text']   = df[title_col] + ' ' + df[text_col]
        return df[['document_id', 'ReviewTitle', 'full_text']].dropna(subset=['full_text'])

    def load_thucnews(self, filepath: str) -> pd.DataFrame:
        """
        Loads THUCNews CSV.

        FIX 5: Added column validation. The original assumed 'id', 'title',
        'content' silently; a missing column produced a cryptic KeyError deep
        in the pipeline.
        Expected columns: id, title, content
        """
        print(f"Loading THUCNews from: {filepath}")
        df = pd.read_csv(filepath)

        required = {'id', 'title', 'content'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"THUCNews CSV missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        df['document_id'] = 'thucnews_' + df['id'].astype(str)
        df['full_text']   = df['title'].fillna('') + ' ' + df['content'].fillna('')
        return df[['document_id', 'title', 'full_text']].dropna(subset=['full_text'])

    def load_sogou(self, filepath: str) -> pd.DataFrame:
        """
        Loads Sogou News CSV.

        FIX 5: Added column validation.
        Expected columns: url_id, contenttitle, content
        """
        print(f"Loading Sogou News from: {filepath}")
        df = pd.read_csv(filepath)

        required = {'url_id', 'contenttitle', 'content'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Sogou News CSV missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        df['document_id'] = 'sogou_' + df['url_id'].astype(str)
        df['full_text']   = df['contenttitle'].fillna('') + ' ' + df['content'].fillna('')
        return df[['document_id', 'contenttitle', 'full_text']].dropna(subset=['full_text'])

    def load_miracl_zh(self, corpus_path: str) -> pd.DataFrame:
        """
        Loads the MIRACL Chinese corpus from JSONL format produced by
        download_miracl_chinese.py.

        JSONL format (one record per line):
            {"_id": "...", "title": "...", "text": "..."}

        document_id is prefixed with 'miracl_' to match the prefix applied
        by download_miracl_chinese.py when building qrel lookup keys in
        build_evaluation_dataset_miracl(). Both must use the same prefix
        or chunk_id matching will silently produce zero hits.

        Note: Unlike the aviation and news loaders, this loader does NOT
        return a 'ReviewTitle' or query column. MIRACL evaluation uses real
        qrels from queries.jsonl/qrels/dev.tsv rather than the title-as-query
        proxy. The evaluation dataset is built separately by
        build_evaluation_dataset_miracl() in main_pipeline.py.
        """
        print(f"Loading MIRACL Chinese corpus from: {corpus_path}")

        records = []
        with open(corpus_path, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    records.append({
                        'document_id': f'miracl_{doc["_id"]}',
                        'full_text':   (doc.get('title', '') + ' ' + doc['text']).strip(),
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  WARNING: Skipping malformed line {line_num}: {e}")

        if not records:
            raise RuntimeError(
                f"No records loaded from '{corpus_path}'. "
                "Check that the file is valid JSONL produced by "
                "download_miracl_chinese.py."
            )

        df = pd.DataFrame(records)
        df = df[df['full_text'].str.strip().astype(bool)]   # drop empty text rows
        print(f"  Loaded {len(df):,} passages.")
        return df

    def load_mr_tydi(self, corpus_path: str, language: str) -> pd.DataFrame:
        """
        Loads a Mr. TyDi corpus from JSONL format produced by download_mr_tydi.py.

        Supported languages: 'ja' (Japanese), 'th' (Thai).

        JSONL format (one record per line, same as MIRACL after normalisation):
            {"_id": "...", "title": "...", "text": "..."}

        document_id is prefixed with 'mr_tydi_{language}_' (e.g. 'mr_tydi_ja_')
        to match the id_prefix used in build_evaluation_dataset_qrels() when
        looking up chunk_ids from qrel corpus_id values. Both sides must use
        the same prefix or chunk_id matching silently produces zero hits.
        """
        lang_name = {'ja': 'Japanese', 'th': 'Thai'}.get(language, language)
        print(f"Loading Mr. TyDi {lang_name} corpus from: {corpus_path}")

        id_prefix = f'mr_tydi_{language}_'
        records = []
        with open(corpus_path, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    records.append({
                        'document_id': f'{id_prefix}{doc["_id"]}',
                        'full_text':   (doc.get('title', '') + ' ' + doc['text']).strip(),
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  WARNING: Skipping malformed line {line_num}: {e}")

        if not records:
            raise RuntimeError(
                f"No records loaded from '{corpus_path}'. "
                "Check that the file is valid JSONL produced by download_mr_tydi.py."
            )

        df = pd.DataFrame(records)
        df = df[df['full_text'].str.strip().astype(bool)]
        print(f"  Loaded {len(df):,} passages.")
        return df

    def load_medweb(self, filepath: str, language: str) -> pd.DataFrame:
        """
        Loads one language partition of the MedWeb multilingual medical
        social media dataset.

        MedWeb is a parallel corpus — the same patient symptom reports are
        available in English ('en'), Japanese ('ja'), and Chinese ('zh').
        Document IDs encode the language as a suffix: '1en', '1ja', '1zh'.
        Documents with the same number are parallel translations, enabling
        true cross-lingual retrieval evaluation.

        Parameters
        ----------
        filepath : str
            Path to medweb_rag_all_fixed.csv (all languages in one file).
        language : str
            Language partition to load: 'en', 'ja', or 'zh'.

        Returns
        -------
        pd.DataFrame with columns: document_id, ReviewTitle, full_text.
            ReviewTitle is used as the retrieval query (title-as-query proxy).
            full_text = ReviewTitle + '. ' + ReviewText.

        IMPORTANT — chunk_size warning
        --------------------------------
        MedWeb texts are very short (English avg ~12 words, Japanese/Chinese
        avg ~10-15 morphemes/tokens). The paper's runs use chunk_size=512
        (and chunk_size_zh/ja=512), so every MedWeb document is a single
        chunk:

            loader = UniversalRAGLoader(
                chunk_size=512, chunk_size_zh=512, chunk_size_ja=512, overlap=73
            )

        (camera_ready_extras.make_loader(); main_pipeline1_patched.py
        --chunk_size 512 --chunk_size_ja 512 --chunk_size_zh 512).
        """
        if language not in ('en', 'ja', 'zh'):
            raise ValueError(
                f"MedWeb language must be 'en', 'ja', or 'zh'. Got: '{language}'"
            )

        print(f"Loading MedWeb ({language.upper()}) from: {filepath}")

        # MedWeb files are comma-separated CSV.
        # utf-8-sig handles BOM that Excel sometimes writes on Windows.
        try:
            df = pd.read_csv(filepath, sep=',', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, sep=',', encoding='utf-8-sig')

        required = {'document_id', 'ReviewTitle', 'ReviewText'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"MedWeb CSV missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        # Filter to the requested language by document_id suffix
        df = df[df['document_id'].astype(str).str.endswith(language)].copy()

        if df.empty:
            raise RuntimeError(
                f"No rows found for language='{language}' in '{filepath}'. "
                f"Document IDs should end with the language code (e.g. '1{language}')."
            )

        df['document_id'] = 'medweb_' + df['document_id'].astype(str)
        df['ReviewTitle'] = df['ReviewTitle'].fillna('')
        df['ReviewText']  = df['ReviewText'].fillna('')
        df['full_text']   = df['ReviewTitle'] + '. ' + df['ReviewText']

        print(f"  Loaded {len(df):,} {language.upper()} documents.")
        return df[['document_id', 'ReviewTitle', 'full_text']].dropna(subset=['full_text'])

    def load_beir(self, corpus_path: str, prefix: str = "nfcorpus_") -> pd.DataFrame:
        """
        Loads NFCorpus from BEIR JSONL format.
        Documents are biomedical MedLine abstracts, avg ~217 words.
        Much longer than MedWeb — tests HHI exclusivity on richer text.
        """
        # Derive a readable dataset label from the path (this loader is
        # generic across all BEIR-format datasets — nfcorpus, chatdoctor,
        # cmedqa, trec_covid — so a hardcoded "NFCorpus" label was misleading
        # in the logs, e.g. printing "Loading NFCorpus" above Chinese cmedqa
        # data).
        import os as _os
        _dataset_label = _os.path.basename(_os.path.dirname(corpus_path)) or "BEIR"
        print(f"Loading BEIR dataset '{_dataset_label}' from: {corpus_path}")
        records = []
        with open(corpus_path, encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                records.append({
                    'document_id': f'{prefix}{doc["_id"]}',
                    'full_text': (doc.get('title','') + ' ' + doc['text']).strip(),
                })
        df = pd.DataFrame(records)
        df = df[df['full_text'].str.strip().astype(bool)]
        print(f"  Loaded {len(df):,} passages.")
        return df

    def clean_and_tokenize(self, text: str, language: str) -> List[str]:
        """
        Routes text through the language-appropriate NLP pipeline.

        English : NLTK word_tokenize → lowercase → alphabetic filter →
                  stopword removal → lemmatisation → length > 2
        Chinese : jieba.lcut → stopword removal → length >= 2
                  FIX 6: changed len >= 1 to len >= 2. Single Chinese
                  characters are almost always ambiguous out of context (e.g.
                  '机' could mean aircraft, machine, or opportunity). Filtering
                  them reduces vocabulary noise without discarding meaningful
                  two-character technical terms like '液压' (hydraulic).
        """
        if not isinstance(text, str):
            return []

        if language == 'en':
            # BUGFIX (#15): NLTK's word_tokenize requires the 'punkt' resource,
            # which repeatedly failed to resolve in some environments due to a
            # version-layout mismatch (loader requests punkt/PY3/*.pickle while
            # newer nltk installs a different layout). Since the next filter
            # keeps only .isalpha() tokens anyway, a regex word-boundary
            # tokenizer was expected to produce the same downstream tokens.
            # It does not (docs/FIXES.md #30): contractions and hyphenated
            # words split differently, and English results change. We try
            # punkt first and fall back to regex only if punkt is unavailable;
            # results match the paper only on machines where punkt loads.
            try:
                tokens = word_tokenize(text.lower())
            except (LookupError, OSError):
                # LookupError: punkt missing. OSError: punkt/PY3_tab layout bug
                # (nltk/nltk#3305). Either way, fall back to a resource-free
                # regex tokenizer — identical BoW output after the .isalpha()
                # filter below (see FIX 15).
                import re as _re
                tokens = _re.findall(r"[A-Za-z]+", text.lower())
            return [
                self.lemmatizer.lemmatize(word) for word in tokens
                if word.isalpha()
                and word not in self.en_stopwords
                and len(word) > 2
            ]
        elif language == 'zh':
            tokens = jieba.lcut(text)
            return [
                word for word in tokens
                if word.strip()
                and word not in self.zh_stopwords
                and len(word) >= 2
            ]
        elif language == 'ja':
            if self._ja_tagger is None:
                raise ImportError(
                    "Japanese tokenization requires fugashi and ipadic.\n"
                    "Install with: pip install fugashi ipadic"
                )
            tokens = [word.surface for word in self._ja_tagger(text)]
            return [
                word for word in tokens
                if word.strip()
                and word not in self.ja_stopwords
                and len(word) >= 2
            ]
        elif language == 'th':
            if self._th_tokenize is None:
                raise ImportError(
                    "Thai tokenization requires pythainlp.\n"
                    "Install with: pip install pythainlp"
                )
            tokens = self._th_tokenize(text, engine='newmm')
            return [
                word for word in tokens
                if word.strip()
                and word not in self.th_stopwords
                and len(word) >= 2
            ]
        else:
            raise ValueError(
                f"Unsupported language '{language}'. "
                "Use 'en', 'zh', 'ja', or 'th'."
            )

    def chunk_document(self, text: str, doc_id: str, language: str) -> List[Dict]:
        """
        Sliding-window chunker. Uses language-appropriate chunk size.

        FIX 3 (chunk size): Chinese chunks now use self.chunk_size_zh (default:
        chunk_size * 2) rather than chunk_size. Because jieba tokens average
        ~1.5 characters, chunk_size=150 jieba tokens ≈ 225 characters ≈ ~45
        English words — far less content than a 150-word English chunk. Doubling
        the token count brings the two languages to roughly comparable semantic
        density per chunk, making cross-lingual MRR results more meaningful.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        # Map language → (word splitter, chunk size, join separator)
        if language == 'en':
            words = text.split()
            cs    = self.chunk_size
            sep   = ' '
        elif language == 'zh':
            words = jieba.lcut(text)
            cs    = self.chunk_size_zh
            sep   = ''
        elif language == 'ja':
            if self._ja_tagger is None:
                raise ImportError(
                    "Japanese tokenization requires fugashi and ipadic.\n"
                    "Install with: pip install fugashi ipadic"
                )
            words = [w.surface for w in self._ja_tagger(text)]
            cs    = self.chunk_size_ja
            sep   = ''
        elif language == 'th':
            if self._th_tokenize is None:
                raise ImportError(
                    "Thai tokenization requires pythainlp.\n"
                    "Install with: pip install pythainlp"
                )
            words = self._th_tokenize(text, engine='newmm')
            cs    = self.chunk_size_th
            sep   = ''
        else:
            raise ValueError(
                f"Unsupported language '{language}'. "
                "Use 'en', 'zh', 'ja', or 'th'."
            )

        stride = cs - self.overlap
        chunks = []

        for i in range(0, len(words), stride):
            chunk_words = words[i: i + cs]
            if len(chunk_words) < 3: #max(1, int(cs * 0.2)):
                continue

            chunk_text = sep.join(chunk_words)
            chunks.append({
                'doc_id':       doc_id,
                'chunk_id':     f'{doc_id}_chunk_{len(chunks)}',
                'text':         chunk_text,
                'clean_tokens': self.clean_and_tokenize(chunk_text, language),
            })

        return chunks

    # ==========================================
    # 3. MAIN PROCESSING PIPELINE
    # ==========================================

    def process_corpus(
        self,
        df: pd.DataFrame,
        language: str,
    ) -> Tuple[List[Dict], List[List[Tuple[int, int]]], List[int]]:
        """
        Chunks the corpus, builds the vocabulary, and returns BoW for Lex-TM.

        Returns
        -------
        all_chunks : list of dict
            Chunk metadata including 'doc_id', 'chunk_id', 'text',
            'clean_tokens'. Pass to LexTMRouter.index_corpus() as raw texts.
        corpus_bow : list of list of (int, int)
            Bag-of-words representation for each chunk, using vocabulary IDs.
            Pass directly to LexTMLdaModel.fit().
        doc_assignments : list of int
            FIX 7: Maps each chunk index to an integer document ID. Required by
            the revised LexTMLdaModel.fit(doc_assignments=...) so that HHI is
            computed at document scope (not chunk scope) as the paper specifies.
            Build this here because the loader already tracks doc_id per chunk.

            Usage in main_pipeline.py:
                all_chunks, corpus_bow, doc_assignments = loader.process_corpus(df, lang)
                lex_tm.fit(..., doc_assignments=doc_assignments)
        """
        print(f"Tokenising and chunking {len(df)} documents (language: {language.upper()})...")
        all_chunks: List[Dict] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Chunking {language.upper()}", unit="doc"):
            doc_chunks = self.chunk_document(row['full_text'], row['document_id'], language)
            all_chunks.extend(doc_chunks)

        print(f"Generated {len(all_chunks)} overlapping chunks.")

        # FIX 7: Build doc_assignments in parallel with chunking.
        # Map string doc_ids to compact integer IDs for Numba compatibility.
        doc_id_to_int: Dict[str, int] = {}
        doc_assignments: List[int] = []
        for chunk in all_chunks:
            did = chunk['doc_id']
            if did not in doc_id_to_int:
                doc_id_to_int[did] = len(doc_id_to_int)
            doc_assignments.append(doc_id_to_int[did])

        # FIX 4: Vocabulary frequency filtering (min_df).
        # Original included every token seen at least once, producing an inflated
        # vocabulary dominated by noise tokens. Filtering to min_df >= 2 removes
        # hapax legomena while keeping rare but recurring technical terms.
        # A token must appear in at least min_df CHUNKS (not total occurrences)
        # to enter the vocabulary, matching standard document-frequency semantics.
        print(f"Counting token document frequencies (min_df={self.min_df})...")
        token_df_counter: Counter = Counter()
        for chunk in all_chunks:
            # Count each token once per chunk (document frequency, not term freq)
            token_df_counter.update(set(chunk['clean_tokens']))

        valid_tokens = {
            token for token, df_count in token_df_counter.items()
            if df_count >= self.min_df
        }

        print(f"Vocabulary: {len(token_df_counter)} raw tokens → "
              f"{len(valid_tokens)} after min_df={self.min_df} filtering.")

        self.token2id = {token: idx for idx, token in enumerate(sorted(valid_tokens))}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        self.vocab_size = len(self.token2id)

        if self.vocab_size == 0:
            raise RuntimeError(
                "Vocabulary is empty after filtering. "
                "Lower min_df or check that clean_and_tokenize() returns tokens."
            )

        # Build BoW: only include tokens that passed min_df filtering
        corpus_bow: List[List[Tuple[int, int]]] = []
        empty_chunk_count = 0
        for chunk in all_chunks:
            word_counts: Dict[int, int] = {}
            for token in chunk['clean_tokens']:
                if token in self.token2id:   # skip filtered-out tokens
                    word_id = self.token2id[token]
                    word_counts[word_id] = word_counts.get(word_id, 0) + 1
            if not word_counts:
                empty_chunk_count += 1
            corpus_bow.append(list(word_counts.items()))

        if empty_chunk_count > 0:
            print(
                f"WARNING: {empty_chunk_count} chunks produced empty BoW after "
                "vocabulary filtering. These will be treated as empty documents "
                "in Gibbs sampling. Consider lowering min_df or reviewing "
                "your stopword list."
            )

        print(f"Pipeline complete. "
              f"Vocabulary size: {self.vocab_size} | "
              f"Chunks: {len(all_chunks)} | "
              f"Documents: {len(doc_id_to_int)}")

        return all_chunks, corpus_bow, doc_assignments
