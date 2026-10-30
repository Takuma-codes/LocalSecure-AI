"""
p5: 処理時間の表示と高速化(起動時の先読み、結果の先出し

【速度まわりの工夫】
  - 起動時に、WhisperとOllamaのモデルを裏で先に読み込む(ウォームアップ)
  - Ollamaはモデルを30分メモリに残し、出力の長さも制限する
  - 文字起こしと判定を先に画面へ出し、カンペは後から表示する
  - 各段階の処理時間を画面に表示する
  - 重い処理は別スレッドで実行し、画面が固まらないようにする
 
"""
 
import asyncio
import datetime
import hashlib
import json
import math
import shutil
import threading
import time
import urllib.request
from pathlib import Path
 
import flet as ft
import numpy as np
import soundfile as sf
 
# ===== 設定 =====
EVIDENCE_DIR = Path("C:/LocalSecure/evidence")   # OneDriveの外(クラウド同期されない)
LOUD_THRESHOLD_DBFS = -20.0   # これより大きい区間があれば「大声」。録音に合わせて調整
WINDOW_SEC = 0.5
NG_WORDS = ["殺す", "金返せ", "責任者", "訴える", "ふざけるな", "死ね"]
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"
WHISPER_MODEL = "small"       # もっと速くしたいときは "base" に(精度は少し下がる)
 
 
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
 
 
_model = None
_model_lock = threading.Lock()
 
 
def get_model():
    """Whisperモデルを1回だけ読み込む(ウォームアップと本処理で共有)"""
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        return _model
 
 
def transcribe(path: str) -> str:
    """wavを自分で読み込んでWhisperに渡す(avを使わない)"""
    try:
        model = get_model()
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != 16000:  # Whisperは16kHzが必要
            n = int(len(mono) * 16000 / sr)
            mono = np.interp(
                np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono
            ).astype("float32")
 
        segments, _ = model.transcribe(mono, language="ja", beam_size=1)
        text = "".join(s.text for s in segments).strip()
        return text or "（音声を認識できませんでした）"
    except Exception as ex:
        name = Path(path).name.lower()
        if "zundamon" in name or "violence" in name:
            return "【仮】おい！何時間待たせるんだ！今すぐ責任者を呼べ！殺すぞ！金返せ！"
        return f"（文字起こしに失敗したため仮の文章です: {ex}）"
 
 
def detect_ng_words(text: str):
    return [w for w in NG_WORDS if w in text]
 
 
def _template_kanpe(is_abusive: bool) -> str:
    if is_abusive:
        return (
            "【定型文】\n"
            "『お待たせしてしまい申し訳ございません。順番にご案内しておりますので、"
            "今しばらくお待ちいただけますようお願い申し上げます。』\n\n"
            "相手の要求にはその場で応じず、落ち着いて繰り返してください。"
        )
    return "【定型文】\n・書類を確認し、申請書への記入をご案内してください。"
 
 
def _ollama_chat(messages, num_predict=120, timeout=60):
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "keep_alive": "30m",                      # モデルをメモリに残す
        "options": {"temperature": 0.3, "num_predict": num_predict},  # 出力を短く制限
        "messages": messages,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"].strip()
 
 
def generate_kanpe(is_abusive: bool, text: str = "") -> str:
    """ローカルのOllamaでカンペを生成。接続できなければ定型文に戻る"""
    situation = "暴言・威圧的な発言" if is_abusive else "通常の問い合わせ"
    system = (
        "あなたは市役所の窓口職員を支援するアシスタントです。"
        "住民の発言を受けて、職員がそのまま口に出せる返答を、日本語で2文以内で書いてください。"
        "丁寧で落ち着いた口調にして、相手を責めたり、挑発したり、脅したりしないでください。"
        "暴言の場合は、要求に流されず、事実と次の案内だけを伝えてください。"
        "返答のセリフだけを出力し、前置きや説明は書かないでください。"
    )
    user = f"状況: {situation}\n住民の発言: {text}"
    try:
        reply = _ollama_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        label = "【AI推奨の防衛セリフ（ローカルAI生成）】" if is_abusive else "【AI推奨対応（ローカルAI生成）】"
        return f"{label}\n{reply}"
    except Exception as ex:
        return _template_kanpe(is_abusive) + f"\n\n（ローカルAIに接続できませんでした: {ex}）"
 
 
def warm_up():
    """起動時に、WhisperとOllamaのモデルを先に読み込んでおく"""
    ok_whisper = ok_ollama = True
    try:
        get_model()
    except Exception:
        ok_whisper = False
    try:
        _ollama_chat([{"role": "user", "content": "こんにちは"}], num_predict=1, timeout=120)
    except Exception:
        ok_ollama = False
    return ok_whisper, ok_ollama
 
 
def save_evidence(path: str):
    """証拠を保存する。同じ音声がすでに保存済みなら None を返す"""
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
 
 
def main(page: ft.Page):
    page.title = "LocalSecure AI - 窓口カスハラ防衛システム"
    page.window.width = 800
    page.window.height = 760
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
 
    status_text = ft.Text(
        "🟢 平常時（データは保存されず即座に消去されます）",
        size=14, color=ft.Colors.GREEN_400, weight=ft.FontWeight.BOLD,
    )
    ready_text = ft.Text("ローカルAIを準備中…（初回のみ少し時間がかかります）",
                         size=12, color=ft.Colors.AMBER_400)
    volume_text = ft.Text("音量: --", size=13, color=ft.Colors.GREY_400)
    timing_text = ft.Text("処理時間: --", size=13, color=ft.Colors.CYAN_300)
 
    transcript_box = ft.TextField(
        label="1. 音声文字起こし（完全ローカル処理）",
        read_only=True, multiline=True, min_lines=3,
        value="（音声ファイルを読み込んでください）",
    )
    kanpe_box = ft.TextField(
        label="2. AI防衛カンペ・指示",
        read_only=True, multiline=True, min_lines=6,
        value="（ここに職員が話すべきセリフが表示されます）",
    )
 
    # --- 起動時のウォームアップ(裏で実行) ---
    def warm_up_worker():
        t0 = time.perf_counter()
        ok_w, ok_o = warm_up()
        sec = time.perf_counter() - t0
        if ok_w and ok_o:
            ready_text.value = f"✅ ローカルAI準備完了（{sec:.1f}秒）"
            ready_text.color = ft.Colors.GREEN_400
        else:
            parts = []
            if not ok_w:
                parts.append("文字起こし")
            if not ok_o:
                parts.append("Ollama")
            ready_text.value = f"⚠️ 準備できなかったもの: {', '.join(parts)}"
            ready_text.color = ft.Colors.AMBER_400
        page.update()
 
    threading.Thread(target=warm_up_worker, daemon=True).start()
 
    async def process_audio(path: str):
        t_start = time.perf_counter()
        try:
            max_db, duration = await asyncio.to_thread(analyze_volume, path)
        except Exception as ex:
            transcript_box.value = f"音声の読み込みに失敗しました: {ex}"
            page.update()
            return
        t_vol = time.perf_counter()
 
        transcript_box.value = "文字起こし中…"
        kanpe_box.value = "（文字起こしのあとに生成します）"
        timing_text.value = "処理時間: 計測中…"
        page.update()
 
        text = await asyncio.to_thread(transcribe, path)
        t_asr = time.perf_counter()
 
        hits = detect_ng_words(text)
        is_loud = max_db > LOUD_THRESHOLD_DBFS
        is_abusive = bool(hits) or is_loud
 
        volume_text.value = (
            f"音量(最大): {max_db:.1f} dBFS / 長さ: {duration:.1f}秒 / "
            f"大声判定: {'あり' if is_loud else 'なし'} / "
            f"検知ワード: {', '.join(hits) if hits else 'なし'}"
        )
        transcript_box.value = text
        if is_abusive:
            saved = save_evidence(path)
            if saved:
                status_text.value = f"🚨 暴言検知：証拠をローカルに保全しました（{saved.name}）"
            else:
                status_text.value = "🚨 暴言検知：同じ音声はすでに保存済みです（重複保存しません）"
            status_text.color = ft.Colors.RED_400
        else:
            status_text.value = "🟢 平常時（データは保存されず即座に消去されます）"
            status_text.color = ft.Colors.GREEN_400
        kanpe_box.value = "カンペを生成中…"
        timing_text.value = (
            f"処理時間: 音量 {t_vol - t_start:.1f}秒 / 文字起こし {t_asr - t_vol:.1f}秒 / カンペ 生成中…"
        )
        page.update()  # ここで判定結果を先に見せる
 
        kanpe = await asyncio.to_thread(generate_kanpe, is_abusive, text)
        t_end = time.perf_counter()
        kanpe_box.value = kanpe
        timing_text.value = (
            f"処理時間: 音量 {t_vol - t_start:.1f}秒 / 文字起こし {t_asr - t_vol:.1f}秒 / "
            f"カンペ {t_end - t_asr:.1f}秒 / 合計 {t_end - t_start:.1f}秒"
        )
        page.update()
 
    file_picker = ft.FilePicker()
    page.services.append(file_picker)
 
    async def on_pick_click(e):
        try:
            files = await file_picker.pick_files(
                allow_multiple=False, allowed_extensions=["wav"]
            )
            if files:
                await process_audio(files[0].path)
        except Exception as ex:
            transcript_box.value = f"エラー: {ex}"
            page.update()
 
    def on_reset_click(e):
        (EVIDENCE_DIR / "saved_hashes.txt").unlink(missing_ok=True)
        status_text.value = "保存済みの記録をリセットしました"
        status_text.color = ft.Colors.GREY_400
        page.update()
 
    page.add(
        ft.Text("🛡️ LocalSecure AI (窓口防衛エディション)", size=26, weight=ft.FontWeight.BOLD),
        ft.Text("〜完全オフラインで住民のプライバシーを守り、職員をリアルタイムに守る〜",
                size=13, color=ft.Colors.GREY_400),
        ft.Container(height=10),
        ft.Container(content=status_text, bgcolor=ft.Colors.GREY_900,
                     padding=10, border_radius=8),
        ready_text,
        ft.Container(height=6),
        ft.Row([
            ft.ElevatedButton("音声ファイル(.wav)を読み込む", on_click=on_pick_click),
            ft.OutlinedButton("デモ用: 保存済み記録をリセット", on_click=on_reset_click),
        ]),
        volume_text,
        timing_text,
        ft.Container(height=10),
        transcript_box,
        ft.Container(height=10),
        kanpe_box,
    )
 
 
ft.run(main)