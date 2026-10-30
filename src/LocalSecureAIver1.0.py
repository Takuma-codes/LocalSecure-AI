"""
- 設定画面に「AIカンペ（通常応答）自動生成」スイッチを追加。
- OFF時：🟢正常会話は定型文を即時表示（0秒応答）で爆速化。
- 🚨暴言・威圧（赤判定）時は、スイッチがOFFであっても強制的にAIカンペが起動して防衛セリフを生成。
- 暴言判定が通算2回に達した際、ローカルのWAVファイルから「ずんだもん（暴言はやめてほしいのだ）」の警告音声を緊急自動再生。
- カンペ生成LLMを 1.5b に軽量化し、プロンプト計算量を削減して応答速度を大幅向上。
"""

import asyncio
import collections
import csv
import datetime
import hashlib
import json
import math
import os
import re
import queue
import shutil
import tempfile
import threading
import time
import urllib.request
import winsound  # ★ ずんだもん音声再生用
from pathlib import Path

import flet as ft
import numpy as np
import soundfile as sf

# ===== 設定 =====
EVIDENCE_DIR = Path("C:/LocalSecure/evidence")
ARCHIVE_DIR = Path("C:/LocalSecure/archive")
TEMP_DIR = Path("C:/LocalSecure/temp")
WINDOW_SEC = 0.5
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:1.5b"       # カンペ生成用（1.5bで速度重視）
FAST_OLLAMA_MODEL = "qwen2.5:1.5b"  # 判定用（1.5b）
OLLAMA_NUM_CTX = 2048               # 文脈ウィンドウ縮小で高速化
MANUAL_STORE_CHARS = 30000
MANUAL_EXCERPT_CHARS = 500          # 読み込み最適化
MANUAL_CHUNK_CHARS = 300
MANUAL_MIN_SCORE = 0.5
PROMPT_CHAR_BUDGET = 2000
WHISPER_MODEL = "small"             # 高精度 Whisper small

# ★ ずんだもん警告音声ファイルのパス（実行ファイルと同階層）
ZUNDA_WAV_PATH = Path(__file__).parent / "zundamon_warning.wav"

# ===== マイク入力の設定 =====
MIC_SAMPLE_RATE = 16000
MIC_BLOCK_SEC = 0.05
MIC_SPEECH_DBFS = -45.0
MIC_SILENCE_SEC = 0.5               # 沈黙検知 0.5秒
MIC_MIN_SPEECH_SEC = 0.3
MIC_MAX_UTTERANCE_SEC = 6.0          # 文脈保持 6.0秒
MIC_PREROLL_SEC = 0.3

# ===== 利用シーンごとの設定 =====
SCENES = {
    "市役所の窓口": {
        "ng_words": ["殺す", "金返せ", "責任者", "訴える", "ふざけるな", "死ね"],
        "loud_dbfs": -20.0,
        "role": "市役所の窓口職員",
        "guide": "公的機関として、公平で丁寧な言葉づかいにする。手続きの内容や可否は断定せず、確認する旨を伝える。",
        "examples": [
            ("暴言・威圧的な発言", "何時間待たせるんだ、今すぐ責任者を呼べ",
             "お待たせしてしまい、ご不快な思いをおかけして申し訳ございません。状況を上の者に伝えますので、少々お待ちいただけますでしょうか。"),
            ("通常の問い合わせ", "更新の手続きに来ました",
             "ご来庁ありがとうございます。手続きの内容を確認いたしますので、少々お待ちください。"),
        ],
        "template_abusive": "『お待たせしてしまい申し訳ございません。順番にご案内しておりますので、今しばらくお待ちいただけますようお願い申し上げます。』",
        "template_normal": "・書類を確認し、申請書への記入をご案内してください。",
    },
    "コールセンター": {
        "ng_words": ["殺す", "死ね", "ふざけるな", "訴える", "上司を出せ", "責任者", "土下座"],
        "loud_dbfs": -22.0,
        "role": "コールセンターのオペレーター",
        "guide": "電話口なので、短く落ち着いた言葉にする。お詫びと共感を先に述べ、確認できる範囲の対応を伝える。上席への引き継ぎは、手順に沿って案内する。",
        "examples": [
            ("暴言・威圧的な発言", "ふざけるな、上司を出せ",
             "ご不快な思いをさせてしまい、申し訳ございません。状況を確認のうえ、上席に引き継ぐ手続きをご案内いたします。"),
            ("通常の問い合わせ", "料金の明細を確認したいのですが",
             "お問い合わせありがとうございます。ご契約の内容を確認いたしますので、少々お待ちください。"),
        ],
        "template_abusive": "『ご不快な思いをさせてしまい申し訳ございません。状況を確認し、上席への引き継ぎをご案内いたします。』",
        "template_normal": "・ご契約内容を確認し、明細の確認方法をご案内してください。",
    },
    "店舗の接客": {
        "ng_words": ["殺す", "死ね", "ふざけるな", "訴える", "店長", "金返せ", "土下座", "弁償"],
        "loud_dbfs": -18.0,
        "role": "店舗の接客スタッフ",
        "guide": "対面なので、短く、すぐ口に出せる言葉にする。まずお詫びを述べ、店長への確認や対応の流れを伝える。その場で返金や交換を約束しない。",
        "examples": [
            ("暴言・威圧的な発言", "店長を出せ、金返せ",
             "ご迷惑をおかけして申し訳ございません。店長に状況を確認してまいりますので、少々お待ちください。"),
            ("通常の問い合わせ", "これの色違いはありますか",
             "ありがとうございます。在庫を確認いたしますので、少々お待ちください。"),
        ],
        "template_abusive": "『ご迷惑をおかけして申し訳ございません。店長に確認してまいりますので、少々お待ちください。』",
        "template_normal": "・在庫を確認し、お客様にご案内してください。",
    },
}
DEFAULT_SCENE = "市役所の窓口"


def play_zundamon_file():
    """保存済みずんだもんWAVファイルを直接再生する関数（再生中はマイク入力をブロック）"""
    global is_zundamon_speaking
    try:
        if ZUNDA_WAV_PATH.exists():
            is_zundamon_speaking = True  # 喋りはじめ：ON
            # SND_FILENAMEのみ指定で再生完了まで待機（スレッド内なのでUIは止まりません）
            winsound.PlaySound(str(ZUNDA_WAV_PATH), winsound.SND_FILENAME)
        else:
            print(f"⚠️ ずんだもんの音声ファイルが見つかりません: {ZUNDA_WAV_PATH}")
    except Exception as ex:
        print(f"⚠️ 音声再生エラー: {ex}")
    finally:
        is_zundamon_speaking = False  # 再生終了：OFF


def check_abuse_with_fast_llm(text: str) -> bool:
    cleaned_text = text.strip()
    if len(cleaned_text) < 4 or cleaned_text.startswith("（"):
        return False

    system_prompt = (
        "あなたはカスタマーハラスメント判定器です。"
        "入力された発言が【危険な発言】に該当する場合は「1」、通常のやり取りの場合は「0」と判定してください。\n\n"
        "【危険な発言（1と判定するもの）】\n"
        "- 相手を脅す・困らせる・圧力をかける・問い詰める文脈\n"
        "- 乱暴な言葉遣いでの要求、不当要求、強要（「さっさと〜しろ」「責任者を呼べ」「上司に代われ」「土下座しろ」など）\n"
        "- 一方的な交代要求や横柄な態度・催促\n\n"
        "【安全な発言（0と判定するもの）】\n"
        "- 通常の案内・質問・丁寧な依頼・挨拶\n"
        "- 「暴言はやめてほしいのだ」「〜のだ」といったアシスタント（ずんだもん）の警告・支援音声\n\n"
        "【出力ルール】\n"
        "理由や説明は一切書かず、「1」または「0」の1文字だけを出力してください。"
    )

    body = json.dumps({
        "model": FAST_OLLAMA_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1,
            "num_ctx": 512,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"発言: {cleaned_text}"}
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            res = json.loads(r.read().decode("utf-8"))["message"]["content"].strip()
            return "1" in res
    except Exception:
        return False


def analyze_volume(path: str):
    data, sr = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)
    win = max(int(sr * WINDOW_SEC), 1)
    max_db = -120.0
    for i in range(0, len(mono), win):
        chunk = mono[i:i + win]
        if len(chunk) == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        max_db = max(max_db, 20 * math.log10(rms + 1e-10))
    return max_db, len(mono) / sr


class UtteranceSegmenter:
    def __init__(self, sr: int):
        self.sr = sr
        self.preroll_max = int(sr * MIC_PREROLL_SEC)
        self.silence_limit = int(sr * MIC_SILENCE_SEC)
        self.min_voiced = int(sr * MIC_MIN_SPEECH_SEC)
        self.max_len = int(sr * MIC_MAX_UTTERANCE_SEC)
        self._reset()

    def _reset(self):
        self.in_speech = False
        self._preroll = collections.deque()
        self._preroll_len = 0
        self._chunks = []
        self._len = 0
        self._voiced = 0
        self._silent_run = 0

    def feed(self, block):
        db = 20 * math.log10(float(np.sqrt(np.mean(block ** 2))) + 1e-10)
        voiced = db > MIC_SPEECH_DBFS

        if not self.in_speech:
            if not voiced:
                self._preroll.append(block)
                self._preroll_len += len(block)
                while self._preroll and self._preroll_len - len(self._preroll[0]) >= self.preroll_max:
                    self._preroll_len -= len(self._preroll.popleft())
                return db, None
            self.in_speech = True
            self._chunks = list(self._preroll)
            self._len = self._preroll_len
            self._preroll.clear()
            self._preroll_len = 0

        self._chunks.append(block)
        self._len += len(block)
        if voiced:
            self._voiced += len(block)
            self._silent_run = 0
        else:
            self._silent_run += len(block)

        if self._silent_run >= self.silence_limit or self._len >= self.max_len:
            return db, self._finish()
        return db, None

    def flush(self):
        return self._finish() if self.in_speech else None

    def _finish(self):
        audio = np.concatenate(self._chunks) if self._chunks else None
        voiced = self._voiced
        self._reset()
        if audio is None or voiced < self.min_voiced:
            return None
        return audio


def open_mic_stream(sd, block_q):
    def callback(indata, frames, time_info, status):
        block_q.put(indata[:, 0].copy())

    rates = [MIC_SAMPLE_RATE]
    try:
        default_sr = int(sd.query_devices(kind="input")["default_samplerate"])
        if default_sr != MIC_SAMPLE_RATE:
            rates.append(default_sr)
    except Exception:
        pass

    last_ex = None
    for sr in rates:
        try:
            return sd.InputStream(
                samplerate=sr, channels=1, dtype="float32",
                blocksize=int(sr * MIC_BLOCK_SEC), callback=callback,
            )
        except Exception as ex:
            last_ex = ex
    raise last_ex


def save_temp_wav(audio, sr: int) -> str:
    fd, path = tempfile.mkstemp(prefix="localsecure_mic_", suffix=".wav")
    os.close(fd)
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


def run_mic_segmenter(block_q, sr, loop, utt_q, on_level, on_error):
    seg = UtteranceSegmenter(sr)
    last_ui = 0.0

    def hand_over(audio):
        if audio is not None:
            loop.call_soon_threadsafe(utt_q.put_nowait, save_temp_wav(audio, sr))

    try:
        while True:
            block = block_q.get()
            if block is None:
                break
            db, audio = seg.feed(block)
            hand_over(audio)
            now = time.perf_counter()
            if now - last_ui >= 0.15:
                last_ui = now
                on_level(db, seg.in_speech)
        hand_over(seg.flush())
    except Exception as ex:
        on_error(ex)
    finally:
        loop.call_soon_threadsafe(utt_q.put_nowait, None)


_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        return _model


def transcribe(path: str) -> str:
    try:
        model = get_model()
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != 16000:
            n = int(len(mono) * 16000 / sr)
            mono = np.interp(
                np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono
            ).astype("float32")

        segments, _ = model.transcribe(mono, language="ja", beam_size=1)
        text = "".join(s.text for s in segments).strip()
        return text or "（音声が無音か認識不可でした）"
    except Exception as ex:
        return f"（文字起こしエラー: {ex}）"


def detect_ng_words(text: str, words):
    return [w for w in words if w in text]


def _template_kanpe(is_abusive: bool, scene: dict) -> str:
    body = scene["template_abusive"] if is_abusive else scene["template_normal"]
    return f"【定型文】\n{body}"


def _ollama_chat_stream(messages, callback, num_predict=60, timeout=60):
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": 0.1, "num_predict": num_predict, "num_ctx": OLLAMA_NUM_CTX},
        "messages": messages,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    full_text = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_text += token
                    callback(token)
            except Exception:
                pass
    return full_text.strip()


_PUNCT = re.compile(r"[\s。、，．,.！？!?・「」『』（）()【】\[\]:：;；]")


def _bigrams(text: str):
    t = _PUNCT.sub("", text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def split_manual_chunks(manual_text: str, size: int = MANUAL_CHUNK_CHARS):
    chunks = []
    for block in re.split(r"\n\s*\n", manual_text.strip()):
        block = block.strip()
        if not block:
            continue
        label = ""
        m = re.match(r"(【\d+ページ】)\s*", block)
        if m:
            label = m.group(1)
            block = block[m.end():]
        pieces = []
        for piece in re.split(r"(?<=[。！？\n])", block):
            piece = piece.strip()
            while len(piece) > size:
                pieces.append(piece[:size])
                piece = piece[size:]
            if piece:
                pieces.append(piece)
        cur = ""
        for piece in pieces + [None]:
            if piece is not None and (not cur or len(cur) + len(piece) <= size):
                cur += piece
                continue
            if cur:
                chunks.append(((label + "\n" if label else "") + cur, cur))
            cur = piece or ""
    return chunks


def select_manual_excerpt(manual_text: str, query: str, max_chars: int = MANUAL_EXCERPT_CHARS) -> str:
    manual = manual_text.strip()
    if not manual:
        return ""
    if max_chars < 200:
        return "（プロンプトの長さの都合で、今回はマニュアルを使いません。確認する案内にしてください。）"
    if len(manual) <= max_chars:
        return manual
    chunks = split_manual_chunks(manual)
    grams = [_bigrams(body) for _, body in chunks]
    df = collections.Counter(g for gs in grams for g in gs)
    q = _bigrams(query)
    scores = [sum(1.0 / df[g] for g in q & gs) for gs in grams]
    top = max(scores) if scores else 0
    if top < MANUAL_MIN_SCORE:
        return "（今回の発言に関係しそうな記述は、読み込んだマニュアルから見つかりませんでした。確認する案内にしてください。）"
    picked, used = [], 0
    for i in sorted(range(len(chunks)), key=lambda k: -scores[k]):
        if scores[i] < top * 0.3:
            break
        need = len(chunks[i][0]) + (2 if picked else 0)
        if used + need <= max_chars:
            picked.append(i)
            used += need
    return "\n\n".join(chunks[i][0] for i in sorted(picked))


def build_system_prompt(scene: dict, action_text: str = "", manual_text: str = "") -> str:
    fixed = (
        "【最優先の固定ルール】\n"
        "- 2文以内、80文字程度まで。\n"
        "- 1文目で、お詫びか、要望を受け止める言葉を述べる。\n"
        "- 2文目で、今できる具体的な対応を伝える。\n"
        "- 暴言や脅しがあっても、相手を責めず、要求をその場で約束しない。\n"
        "- 「別途ご相談ください」「他にお手伝いできることはありますか」のような、突き放す言い方や、話をそらす言い方は使わない。\n"
        "- セリフだけを出力し、前置きや説明は書かない。"
    )
    user_policy = action_text.strip()[:200]
    if user_policy:
        optional = (
            "\n【利用者が指定した『取ってほしい行動』】\n"
            f"{user_policy}\n"
            "これは追加の参考方針です。最優先の固定ルールと矛盾する場合は、固定ルールを必ず優先してください。"
        )
    else:
        optional = "\n【利用者が指定した『取ってほしい行動』】\n指定なし。"
    manual = manual_text.strip()[:MANUAL_EXCERPT_CHARS]
    if manual:
        manual_section = (
            "\n【読み込んだ業務マニュアル（参考資料）】\n"
            "次は、マニュアルから今回の発言に近い箇所だけを抜き出したものです。参考にしてください。"
            "資料に書かれていない手続き・条件・約束を推測で作らず、不明な場合は確認する案内にしてください。\n"
            f"{manual}\n【マニュアルここまで】"
        )
    else:
        manual_section = "\n【業務マニュアル】読み込みなし。マニュアルに基づく案内はできません。"
    return (
        f"あなたは{scene['role']}を支援するアシスタントです。"
        "住民やお客様の発言に対して、職員がそのまま口に出せる返答を日本語で書きます。\n"
        f"この現場の方針: {scene['guide']}\n{manual_section}\n{optional}\n{fixed}"
    )


def generate_kanpe_stream(is_abusive: bool, text: str, scene: dict, action_text: str, manual_text: str, callback):
    situation = "暴言・威圧的な発言" if is_abusive else "通常の問い合わせ"
    examples = []
    for sit, said, reply in scene["examples"]:
        examples.append({"role": "user", "content": f"状況: {sit}\n発言: {said}"})
        examples.append({"role": "assistant", "content": reply})
    last = {"role": "user", "content": f"状況: {situation}\n発言: {text}"}

    base_len = (len(build_system_prompt(scene, action_text, ""))
                + sum(len(m["content"]) for m in examples) + len(last["content"]))
    budget = min(MANUAL_EXCERPT_CHARS, PROMPT_CHAR_BUDGET - base_len)
    excerpt = select_manual_excerpt(manual_text, text, budget)

    messages = [{"role": "system", "content": build_system_prompt(scene, action_text, excerpt)}]
    messages += examples
    messages.append(last)
    try:
        return _ollama_chat_stream(messages, callback)
    except Exception as ex:
        err_msg = f"\n（ローカルAI非接続: {ex}）"
        callback(err_msg)
        return _template_kanpe(is_abusive, scene) + err_msg


def archive_record(path, source, scene_name, duration, max_db, is_loud, hits, is_abusive,
                   text, kanpe, evidence_name):
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    dest = ARCHIVE_DIR / f"audio_{now.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.wav"
    shutil.copy(path, dest)
    log = ARCHIVE_DIR / "log.csv"
    is_new = not log.exists()
    with open(log, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["時刻", "入力元", "シーン", "音声ファイル", "長さ(秒)", "最大音量(dBFS)",
                        "大声判定", "検知ワード", "暴言判定", "文字起こし", "カンペ", "証拠ファイル"])
        w.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), source, scene_name, dest.name,
                    f"{duration:.1f}", f"{max_db:.1f}", "あり" if is_loud else "なし",
                    " ".join(hits), "あり" if is_abusive else "なし", text, kanpe, evidence_name])
    return dest


def save_evidence(path: str):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    index = EVIDENCE_DIR / "saved_hashes.txt"
    saved = index.read_text(encoding="utf-8").split() if index.exists() else []
    if digest in saved:
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = EVIDENCE_DIR / f"evidence_{stamp}.wav"
    shutil.copy(path, dest)
    with open(index, "a", encoding="utf-8") as f:
        f.write(digest + "\n")
    return dest


REASONS_FILE = "evidence_reasons.json"


def load_evidence_reasons() -> dict:
    p = EVIDENCE_DIR / REASONS_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def record_evidence_reason(name: str, hits, is_loud: bool, is_fast_llm: bool, max_db: float, loud_dbfs: float, scene_name: str):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    reasons = load_evidence_reasons()
    reasons[name] = {
        "words": list(hits),
        "loud": bool(is_loud),
        "fast_llm": bool(is_fast_llm),
        "max_db": round(float(max_db), 1),
        "loud_dbfs": float(loud_dbfs),
        "scene": scene_name,
    }
    tmp = EVIDENCE_DIR / (REASONS_FILE + ".tmp")
    tmp.write_text(json.dumps(reasons, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, EVIDENCE_DIR / REASONS_FILE)


def format_evidence_reason(r) -> str:
    if not r:
        return "記録なし（理由を記録する前のファイル）"
    parts = []
    if r.get("words"):
        parts.append("検知ワード: " + "、".join(r["words"]))
    if r.get("loud"):
        parts.append(f"大きな発声: {r['max_db']:.1f} dBFS（基準 {r['loud_dbfs']:.0f} dBFS 超え）")
    if r.get("fast_llm"):
        parts.append("文脈判定（AI判定）")
    return "\n".join(parts) if parts else "記録なし"


def warm_up():
    whisper_ok = False
    ollama_ok = False
    try:
        get_model()
        whisper_ok = True
    except Exception:
        whisper_ok = False

    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
        model_names = [item.get("name", "") for item in data.get("models", [])]
        main_ok = any(name == OLLAMA_MODEL or name.startswith(OLLAMA_MODEL + ":") for name in model_names)
        fast_ok = any(name == FAST_OLLAMA_MODEL or name.startswith(FAST_OLLAMA_MODEL + ":") for name in model_names)
        ollama_ok = main_ok and fast_ok
    except Exception:
        ollama_ok = False

    return whisper_ok, ollama_ok


def main(page: ft.Page):
    page.title = "LocalSecure AI - カスハラ防衛ダッシュボード"
    page.window.width = 1100
    page.window.height = 850
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.bgcolor = "#0f172a"

    state = {"scene": DEFAULT_SCENE, "tab": "dash", "total_count": 0, "abusive_count": 0}

    status_badge = ft.Container(
        padding=ft.Padding(24, 8, 24, 8),
        border_radius=24,
        bgcolor="#166534",
        content=ft.Text("🟢 正常監視中", size=13, color="#f0fdf4", weight=ft.FontWeight.BOLD)
    )

    ready_text = ft.Text("ローカルAIを準備中…", size=12, color="#f59e0b")
    volume_text = ft.Text("音量: --", size=11, color="#94a3b8")
    timing_text = ft.Text("処理時間: --", size=11, color="#38bdf8")
    saved_file_text = ft.Text("保存ファイル: まだありません", size=11, color="#fbbf24", selectable=True)

    stat_total_txt = ft.Text("0 件", size=22, weight=ft.FontWeight.BOLD, color="#f8fafc")
    stat_abusive_txt = ft.Text("0 件", size=22, weight=ft.FontWeight.BOLD, color="#ef4444")
    stat_ratio_txt = ft.Text("0 %", size=22, weight=ft.FontWeight.BOLD, color="#38bdf8")

    def update_stats():
        stat_total_txt.value = f"{state['total_count']} 件"
        stat_abusive_txt.value = f"{state['abusive_count']} 件"
        ratio = (state['abusive_count'] / state['total_count'] * 100) if state['total_count'] > 0 else 0
        stat_ratio_txt.value = f"{ratio:.1f} %"

    card_total = ft.Container(
        expand=True, padding=12, bgcolor="#1e293b", border_radius=12, border=ft.Border.all(1, "#334155"),
        content=ft.Row([
            ft.Column([
                ft.Text("本日処理した音声", size=11, color="#94a3b8"),
                stat_total_txt,
                ft.Text("リアルタイムセッション", size=9, color="#64748b")
            ], expand=True, spacing=2),
            ft.Container(padding=8, bgcolor=ft.Colors.with_opacity(0.1, "#3b82f6"), border_radius=8,
                         content=ft.Icon(ft.Icons.MIC_NONE, color="#3b82f6", size=20))
        ])
    )

    card_abusive = ft.Container(
        expand=True, padding=12, bgcolor="#1e293b", border_radius=12, border=ft.Border.all(1, "#334155"),
        content=ft.Row([
            ft.Column([
                ft.Text("暴言・威圧検知数", size=11, color="#94a3b8"),
                stat_abusive_txt,
                ft.Text("自動証拠保全完了", size=9, color="#64748b")
            ], expand=True, spacing=2),
            ft.Container(padding=8, bgcolor=ft.Colors.with_opacity(0.1, "#ef4444"), border_radius=8,
                         content=ft.Icon(ft.Icons.WARNING_AMBER, color="#ef4444", size=20))
        ])
    )

    card_ratio = ft.Container(
        expand=True, padding=12, bgcolor="#1e293b", border_radius=12, border=ft.Border.all(1, "#334155"),
        content=ft.Row([
            ft.Column([
                ft.Text("検知率", size=11, color="#94a3b8"),
                stat_ratio_txt,
                ft.Text("オフラインAI解析", size=9, color="#64748b")
            ], expand=True, spacing=2),
            ft.Container(padding=8, bgcolor=ft.Colors.with_opacity(0.1, "#38bdf8"), border_radius=8,
                         content=ft.Icon(ft.Icons.DONUT_LARGE, color="#38bdf8", size=20))
        ])
    )

    transcript_box = ft.TextField(
        label="1. 音声文字起こし (Whisper / 爆速レスポンス)",
        read_only=True, multiline=True, min_lines=2, max_lines=3,
        value="（マイク監視を開始するか、音声ファイルを読み込んでください）",
        border_color="#334155", bgcolor="#0f172a", text_size=13
    )

    kanpe_box = ft.TextField(
        label="2. AI防衛指示・生成セリフ (Ollama / Local LLM)",
        read_only=True, multiline=True,
        expand=True,
        value="（ここに職員が話すべき推奨セリフがリアルタイムで表示されます）",
        border_color="#334155", bgcolor="#0f172a", text_size=14
    )

    mic_status = ft.Text("マイク待機中", size=12, color="#94a3b8")
    mic_level_text = ft.Text("入力レベル: --", size=11, color="#64748b")

    process_lock = asyncio.Lock()
    mic = {"stream": None, "block_q": None, "consumer": None, "loop": None}

    manual_state = {"text": "", "name": ""}
    manual_status = ft.Text("マニュアル未読込", size=12, color="#94a3b8")
    manual_dash_text = ft.Text("📄 業務マニュアル: 未読込", size=11, color="#94a3b8")
    action_box = ft.TextField(
        label="追加行動指示 (任意・200字以内)",
        hint_text="例：状況を確認し、受付番号を案内する",
        multiline=True, min_lines=2, max_length=200, border_color="#334155"
    )

    # ★ 設定スイッチ（全会話保存モード: 初期値 OFF）
    save_all_switch = ft.Switch(label="全会話保存モード (log.csv + wav)", value=False)
    kanpe_enable_switch = ft.Switch(
        label="AIカンペ（通常会話）自動生成",
        value=True,
        tooltip="OFFにすると通常時は定型文を即時表示し、暴言検知時（赤）のみAIカンペを緊急生成します。"
    )

    async def process_audio_core(path: str):
        scene = SCENES[state["scene"]]
        action_text = (action_box.value or "")[:200]
        manual_text = manual_state["text"][:MANUAL_STORE_CHARS]
        save_all = bool(save_all_switch.value)
        kanpe_enabled = bool(kanpe_enable_switch.value)
        evidence_name = ""
        t_start = time.perf_counter()

        # 一時保存
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        temp_wav_path = TEMP_DIR / Path(path).name
        shutil.copy(path, temp_wav_path)

        try:
            max_db, duration = await asyncio.to_thread(analyze_volume, str(temp_wav_path))
        except Exception as ex:
            transcript_box.value = f"音声読込エラー: {ex}"
            temp_wav_path.unlink(missing_ok=True)
            page.update()
            return
        t_vol = time.perf_counter()

        text = await asyncio.to_thread(transcribe, str(temp_wav_path))
        t_asr = time.perf_counter()

        transcript_box.value = text
        volume_text.value = f"最大音量: {max_db:.1f} dBFS | 長さ: {duration:.1f}s"
        timing_text.value = f"文字起こし速度: {t_asr-t_vol:.2f}s (判定・処理中...)"
        kanpe_box.value = "（解析中...）"
        page.update()

        # ⚡ タスクA: 高速判定 & ファイル確定処理
        hits = detect_ng_words(text, scene["ng_words"])
        is_loud = max_db > scene["loud_dbfs"]
        is_abusive = bool(hits) or is_loud
        is_fast_llm = False

        if not is_abusive:
            is_fast_llm = await asyncio.to_thread(check_abuse_with_fast_llm, text)
            if is_fast_llm:
                is_abusive = True

        state["total_count"] += 1
        if is_abusive:
            state["abusive_count"] += 1

            # 🚨 暴言検知数が「2回」になった瞬間にずんだもん発動！
            if state["abusive_count"] == 2:
                asyncio.to_thread(play_zundamon_file)

        update_stats()

        if is_abusive:
            # 🚨 暴言・威圧（赤）：証拠フォルダへ保存
            saved = save_evidence(str(temp_wav_path))
            if saved:
                evidence_name = saved.name
                saved_file_text.value = f"保存ファイル: {saved.name}"
                status_badge.content.value = f"🚨 暴言・威圧検知 ({saved.name})"
                try:
                    record_evidence_reason(
                        saved.name, hits, is_loud, is_fast_llm, max_db, scene["loud_dbfs"], state["scene"]
                    )
                except Exception:
                    pass
            else:
                status_badge.content.value = "🚨 暴言・威圧検知 (重複保存スキップ)"
                saved_file_text.value = "保存ファイル: 重複保存スキップ"
            status_badge.bgcolor = "#991b1b"
            status_badge.content.color = "#fef2f2"
            temp_wav_path.unlink(missing_ok=True)
        else:
            # 🟢 正常会話（緑）
            status_badge.bgcolor = "#166534"
            status_badge.content.value = "🟢 正常会話 (データ破棄)" if not save_all else "🟢 正常会話 (保存完了)"
            status_badge.content.color = "#f0fdf4"

            if not save_all:
                temp_wav_path.unlink(missing_ok=True)
                saved_file_text.value = "保存ファイル: まだありません（正常データ破棄完了）"
            else:
                source = "マイク" if Path(path).name.startswith("localsecure_mic_") else Path(path).name
                archived_path = await asyncio.to_thread(
                    archive_record, str(temp_wav_path), source, state["scene"], duration, max_db,
                    is_loud, hits, is_abusive, text, _template_kanpe(False, scene), evidence_name
                )
                temp_wav_path.unlink(missing_ok=True)
                saved_file_text.value = f"保存ファイル: {archived_path.name}"

        page.update()

        # ===== 💬 カンペ出力制御ロジック =====
        if kanpe_enabled or is_abusive:
            label = "【🚨AI緊急防衛指示】\n" if is_abusive else "【AI推奨対応案】\n"
            kanpe_box.value = label
            page.update()

            loop = asyncio.get_running_loop()

            def on_token(token: str):
                def update_ui():
                    kanpe_box.value += token
                    page.update()
                loop.call_soon_threadsafe(update_ui)

            await asyncio.to_thread(
                generate_kanpe_stream, is_abusive, text, scene, action_text, manual_text, on_token
            )
        else:
            kanpe_box.value = _template_kanpe(False, scene)

        t_end = time.perf_counter()
        timing_text.value = f"処理時間: 計 {t_end-t_start:.2f}s (音声 {t_asr-t_start:.2f}s / カンペ {t_end-t_asr:.2f}s)"
        page.update()

    async def process_audio(path: str):
        async with process_lock:
            await process_audio_core(path)

    def show_mic_level(db: float, speaking: bool):
        if mic["stream"] is None:
            return
        mic_level_text.value = f"入力レベル: {db:.0f} dBFS | {'🎙️ 発話検知中' if speaking else '待機中'}"
        page.update()

    def stop_mic():
        stream = mic["stream"]
        if stream is None:
            return
        mic["stream"] = None
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        mic["block_q"].put(None)
        mic_start_btn.disabled = False
        mic_stop_btn.disabled = True
        mic_status.value = "マイク停止"
        mic_status.color = "#94a3b8"
        mic_level_text.value = "入力レベル: --"
        page.update()

    def show_mic_error(ex: Exception):
        def apply():
            stop_mic()
            mic_status.value = f"エラー停止: {ex}"
            mic_status.color = "#ef4444"
            page.update()
        mic["loop"].call_soon_threadsafe(apply)

    async def mic_consumer(utt_q: asyncio.Queue):
        while True:
            path = await utt_q.get()
            if path is None:
                break
            try:
                await process_audio(path)
            except Exception as ex:
                transcript_box.value = f"エラー: {ex}"
                page.update()
            finally:
                Path(path).unlink(missing_ok=True)

    async def on_mic_start_click(e):
        if mic["stream"] is not None:
            return
        try:
            import sounddevice as sd
        except ImportError:
            mic_status.value = "sounddevice が必要です"
            page.update()
            return

        block_q = queue.Queue()
        try:
            stream = open_mic_stream(sd, block_q)
            stream.start()
        except Exception as ex:
            mic_status.value = f"マイクエラー: {ex}"
            page.update()
            return

        loop = asyncio.get_running_loop()
        utt_q = asyncio.Queue()
        mic.update(stream=stream, block_q=block_q, loop=loop)
        threading.Thread(
            target=run_mic_segmenter,
            args=(block_q, int(stream.samplerate), loop, utt_q, show_mic_level, show_mic_error),
            daemon=True,
        ).start()
        mic["consumer"] = asyncio.create_task(mic_consumer(utt_q))

        mic_start_btn.disabled = True
        mic_stop_btn.disabled = False
        mic_status.value = "🎙️ マイク監視アクティブ"
        mic_status.color = "#22c55e"
        page.update()

    mic_start_btn = ft.ElevatedButton("マイク監視開始", icon=ft.Icons.PLAY_ARROW, on_click=on_mic_start_click, bgcolor="#2563eb", color="white")
    mic_stop_btn = ft.OutlinedButton("監視停止", icon=ft.Icons.STOP, on_click=lambda e: stop_mic(), disabled=True)

    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    async def on_pick_click(e):
        files = await file_picker.pick_files(allow_multiple=False, allowed_extensions=["wav"])
        if files:
            await process_audio(files[0].path)

    async def on_manual_pick_click(e):
        try:
            files = await file_picker.pick_files(
                allow_multiple=False, allowed_extensions=["pdf"]
            )
            if not files:
                return
            pdf_path = files[0].path
            try:
                from pypdf import PdfReader
            except ImportError:
                manual_status.value = "PDF読込には pypdf が必要です。ターミナルで pip install pypdf を実行してください。"
                manual_status.color = "#ef4444"
                page.update()
                return

            def extract_pdf_text(path):
                reader = PdfReader(path)
                pages_text = []
                for page_number, pdf_page in enumerate(reader.pages, start=1):
                    page_text = pdf_page.extract_text() or ""
                    if page_text.strip():
                        pages_text.append(f"【{page_number}ページ】\n{page_text}")
                return "\n\n".join(pages_text)

            extracted = await asyncio.to_thread(extract_pdf_text, pdf_path)
            if not extracted.strip():
                manual_state["text"] = ""
                manual_state["name"] = ""
                manual_status.value = "文字を抽出できませんでした。スキャンPDFの場合はOCRが必要です。"
                manual_status.color = "#f59e0b"
                manual_dash_text.value = "📄 業務マニュアル: 未読込"
                manual_dash_text.color = "#94a3b8"
            else:
                manual_state["text"] = extracted[:MANUAL_STORE_CHARS]
                manual_state["name"] = Path(pdf_path).name
                truncated = f"（先頭{MANUAL_STORE_CHARS:,}文字を使用）" if len(extracted) > MANUAL_STORE_CHARS else ""
                manual_status.value = f"PDF読込済み: {manual_state['name']} / 抽出 {len(extracted):,}文字 {truncated}"
                manual_status.color = "#22c55e"
                manual_dash_text.value = f"📄 業務マニュアル使用中: {manual_state['name']}"
                manual_dash_text.color = "#22c55e"
            page.update()
        except Exception as ex:
            manual_status.value = f"PDFの読み込みに失敗しました: {ex}"
            manual_status.color = "#ef4444"
            page.update()

    def on_manual_clear_click(e):
        manual_state["text"] = ""
        manual_state["name"] = ""
        manual_status.value = "マニュアル未読込"
        manual_status.color = "#94a3b8"
        manual_dash_text.value = "📄 業務マニュアル: 未読込"
        manual_dash_text.color = "#94a3b8"
        page.update()

    def on_reset_click(e):
        try:
            (EVIDENCE_DIR / "saved_hashes.txt").unlink(missing_ok=True)
            saved_file_text.value = "保存ファイル: 保存済みの記録をリセットしました（同じ音声も再度保存可能）"
        except Exception as ex:
            saved_file_text.value = f"リセットに失敗しました: {ex}"
        page.update()

    main_dashboard_view = ft.Column([
        ft.Row([card_total, card_abusive, card_ratio], spacing=12),
        ft.Container(height=4),
        ft.Container(
            expand=True,
            padding=16,
            bgcolor="#1e293b",
            border_radius=12,
            border=ft.Border.all(1, "#334155"),
            content=ft.Column([
                ft.Row([
                    ft.Text("リアルタイム音声解析パネル", size=15, weight=ft.FontWeight.BOLD, color="#f8fafc"),
                    status_badge
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(color="#334155", height=12),
                ft.Row([
                    mic_start_btn,
                    mic_stop_btn,
                    ft.ElevatedButton("WAVファイル読込", icon=ft.Icons.UPLOAD_FILE, on_click=on_pick_click),
                    ft.OutlinedButton("デモ用: 保存済み記録をリセット", icon=ft.Icons.REFRESH, on_click=on_reset_click),
                    mic_status,
                ], spacing=10, wrap=True),
                ft.Row([manual_dash_text, saved_file_text], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Row([volume_text, timing_text, mic_level_text], spacing=15),
                ft.Container(height=4),
                transcript_box,
                ft.Container(height=4),
                kanpe_box,
            ], expand=True)
        )
    ], expand=True)

    logs_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("ファイル名", color="#38bdf8")),
            ft.DataColumn(ft.Text("サイズ", color="#38bdf8")),
            ft.DataColumn(ft.Text("更新日時", color="#38bdf8")),
            ft.DataColumn(ft.Text("検知理由", color="#38bdf8")),
        ],
        rows=[]
    )

    def refresh_logs():
        rows = []
        reasons = load_evidence_reasons()
        if EVIDENCE_DIR.exists():
            for p in sorted(EVIDENCE_DIR.glob("*.wav"), key=os.path.getmtime, reverse=True):
                stat = p.stat()
                mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                size_kb = f"{stat.st_size / 1024:.1f} KB"
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(p.name, color="#f8fafc")),
                    ft.DataCell(ft.Text(size_kb, color="#94a3b8")),
                    ft.DataCell(ft.Text(mtime, color="#94a3b8")),
                    ft.DataCell(ft.Container(
                        width=380, padding=ft.Padding(0, 6, 0, 6),
                        content=ft.Text(
                            format_evidence_reason(reasons.get(p.name)), size=12,
                            color="#fca5a5" if p.name in reasons else "#64748b",
                        ),
                    )),
                ]))
        logs_table.rows = rows
        page.update()

    logs_view = ft.Container(
        padding=20,
        bgcolor="#1e293b",
        border_radius=12,
        border=ft.Border.all(1, "#334155"),
        visible=False,
        content=ft.Column([
            ft.Row([
                ft.Text("保全済み証拠・ログファイル一覧", size=18, weight=ft.FontWeight.BOLD),
                ft.ElevatedButton("一覧を更新", icon=ft.Icons.REFRESH, on_click=lambda e: refresh_logs())
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Text(f"保存先ディレクトリ: {EVIDENCE_DIR}", size=12, color="#94a3b8"),
            ft.Divider(color="#334155"),
            ft.Column([ft.Row([logs_table], scroll=ft.ScrollMode.AUTO)], scroll=ft.ScrollMode.AUTO, expand=True)
        ], expand=True)
    )

    scene_dropdown = ft.Dropdown(
        value=DEFAULT_SCENE,
        options=[ft.dropdown.Option(k) for k in SCENES.keys()],
        border_color="#334155"
    )
    def on_scene_change(e):
        val = e.control.value if hasattr(e, "control") and e.control else getattr(e, "data", None)
        if val:
            state.update(scene=val)
    scene_dropdown.on_change = on_scene_change

    settings_view = ft.Container(
        padding=20,
        bgcolor="#1e293b",
        border_radius=12,
        border=ft.Border.all(1, "#334155"),
        visible=False,
        expand=True,
        content=ft.Column([
            ft.Text("システム設定", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(color="#334155"),
            ft.Text("AI動作・レスポンス設定", size=14, weight=ft.FontWeight.BOLD, color="#38bdf8"),
            kanpe_enable_switch,
            ft.Text("OFFにすると日常会話のAI生成をスキップして即時定型文を表示します（🚨暴言判定時は自動起動）。", size=11, color="#94a3b8"),
            ft.Container(height=10),
            ft.Text("利用シーン設定", size=14, weight=ft.FontWeight.BOLD, color="#38bdf8"),
            scene_dropdown,
            ft.Container(height=10),
            ft.Text("組織行動プロンプト設定", size=14, weight=ft.FontWeight.BOLD, color="#38bdf8"),
            action_box,
            ft.Container(height=10),
            ft.Text("業務マニュアル（PDF・任意）", size=14, weight=ft.FontWeight.BOLD, color="#38bdf8"),
            ft.Text("PDF内の文字を抽出し、AIの返信（カンペ）を作るときのご参考資料として使います。", size=11, color="#94a3b8"),
            ft.Row([
                ft.ElevatedButton("PDFマニュアルを読み込む", icon=ft.Icons.UPLOAD_FILE, on_click=on_manual_pick_click),
                ft.OutlinedButton("マニュアルを解除", on_click=on_manual_clear_click),
            ], spacing=12, wrap=True),
            manual_status,
            ft.Container(height=10),
            ft.Text("データアーカイブ設定", size=14, weight=ft.FontWeight.BOLD, color="#38bdf8"),
            save_all_switch,
            ft.Text("ONにすると、正常な会話（緑判定）であってもログとWAV音声をすべて保存します。", size=11, color="#94a3b8"),
        ], scroll=ft.ScrollMode.AUTO, expand=True)
    )

    def nav_change(tab_name):
        state["tab"] = tab_name
        main_dashboard_view.visible = (tab_name == "dash")
        logs_view.visible = (tab_name == "logs")
        settings_view.visible = (tab_name == "settings")
        if tab_name == "logs":
            refresh_logs()
        page.update()

    sidebar = ft.Container(
        width=220,
        bgcolor="#0f172a",
        padding=16,
        border=ft.Border.only(right=ft.BorderSide(1, "#334155")),
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.SECURITY, color="#3b82f6", size=28),
                ft.Text("LocalSecure AI", size=18, weight=ft.FontWeight.BOLD, color="#f8fafc")
            ]),
            ft.Text("完全オフライン防衛システム", size=10, color="#64748b"),
            ft.Divider(color="#334155"),
            ft.TextButton("ダッシュボード", icon=ft.Icons.DASHBOARD, on_click=lambda e: nav_change("dash")),
            ft.TextButton("保存ログ・証拠一覧", icon=ft.Icons.FOLDER, on_click=lambda e: nav_change("logs")),
            ft.TextButton("設定・マニュアル", icon=ft.Icons.SETTINGS, on_click=lambda e: nav_change("settings")),
            ft.Container(expand=True),
            ready_text,
            ft.Text("v16.0 / Zundamon Warning Mode", size=10, color="#475569")
        ])
    )

    header = ft.Container(
        padding=ft.Padding(20, 14, 20, 14),
        bgcolor="#1e293b",
        border=ft.Border.only(bottom=ft.BorderSide(1, "#334155")),
        content=ft.Row([
            ft.Text("カスタマーハラスメント 防衛ダッシュボード", size=18, weight=ft.FontWeight.BOLD),
            ft.Text("クラウド非経由・オンプレミスAI処理", size=12, color="#22c55e")
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
    )

    content_area = ft.Container(
        padding=16,
        expand=True,
        content=ft.Stack([main_dashboard_view, logs_view, settings_view])
    )

    right_panel = ft.Column([header, content_area], expand=True, spacing=0)

    def warm_up_worker():
        ok_w, ok_o = warm_up()
        if ok_w and ok_o:
            ready_text.value = "🟢 AIエンジン準備完了"
            ready_text.color = "#22c55e"
        else:
            ready_text.value = "⚠️ 一部エンジン未起動"
            ready_text.color = "#f59e0b"
        page.update()

    threading.Thread(target=warm_up_worker, daemon=True).start()

    page.add(ft.Row([sidebar, right_panel], expand=True, spacing=0))


ft.app(target=main)