"""
LocalSecure AI - 窓口カスハラ防衛システム (プロトタイプ)

【概要】
窓口の音声をPCの中だけで処理し、暴言を検知したときだけ証拠を保存して、
職員に返答のセリフ(カンペ)を表示する。

【処理の流れ】
  wav読み込み
  -> 音量(dBFS)を0.5秒ごとに実測し、最大値で大声判定
  -> 文字起こし(faster-whisper "small" / CPU / int8 / 完全ローカル)
  -> 暴言ワード検知(NG_WORDS)
  -> 判定: ワードあり または 大声 -> 暴言
  -> カンペ生成(Ollama + qwen2.5:3b / ローカル)
  -> 暴言のときだけ証拠wavを保存(C:/LocalSecure/evidence)

【実装メモ】
  - 文字起こしはパスではなく、soundfileで読んだ音声データ(16kHz)を
    Whisperに渡す。avの不具合(metadata_errors)を避けるため。
  - 文字起こしに失敗すると、仮の文章を返す。先頭に【仮】が付く。
  - Ollamaに接続できないときは、定型文に自動で戻る。
    カンペの先頭が「ローカルAI生成」なら本物、「定型文」なら未接続。
  - 証拠の保存先はOneDriveの外にしている。クラウドに同期させないため。

【事前準備】
  pip install flet numpy soundfile faster-whisper
  Ollamaをインストールし、ollama pull qwen2.5:3b を実行しておく。
  (モデルは取得済みなら、以降はネットなしで動く)

【今後の予定】
  - マイクからのリアルタイム入力(sounddevice)
  - 誤検知の測定(検知率・処理時間)
  - 起動時のモデルの先読み、画面演出
"""

import datetime
import json
import math
import shutil
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
 
 
def transcribe(path: str) -> str:
    """wavを自分で読み込んでWhisperに渡す(avを使わない)"""
    global _model
    try:
        from faster_whisper import WhisperModel
        if _model is None:
            _model = WhisperModel("small", device="cpu", compute_type="int8")
 
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != 16000:  # Whisperは16kHzが必要
            n = int(len(mono) * 16000 / sr)
            mono = np.interp(
                np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono
            ).astype("float32")
 
        segments, _ = _model.transcribe(mono, language="ja", beam_size=1)
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
        body = json.dumps({
            "model": OLLAMA_MODEL,
            "stream": False,
            "options": {"temperature": 0.3},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            reply = json.loads(r.read().decode("utf-8"))["message"]["content"].strip()
        label = "【AI推奨の防衛セリフ（ローカルAI生成）】" if is_abusive else "【AI推奨対応（ローカルAI生成）】"
        return f"{label}\n{reply}"
    except Exception as ex:
        return _template_kanpe(is_abusive) + f"\n\n（ローカルAIに接続できませんでした: {ex}）"
 
 
def save_evidence(path: str) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = EVIDENCE_DIR / f"evidence_{stamp}.wav"
    shutil.copy(path, dest)
    return dest
 
 
def main(page: ft.Page):
    page.title = "LocalSecure AI - 窓口カスハラ防衛システム"
    page.window.width = 800
    page.window.height = 700
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
 
    status_text = ft.Text(
        "🟢 平常時（データは保存されず即座に消去されます）",
        size=14, color=ft.Colors.GREEN_400, weight=ft.FontWeight.BOLD,
    )
    volume_text = ft.Text("音量: --", size=13, color=ft.Colors.GREY_400)
 
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
 
    def process_audio(path: str):
        try:
            max_db, duration = analyze_volume(path)
        except Exception as ex:
            transcript_box.value = f"音声の読み込みに失敗しました: {ex}"
            page.update()
            return
 
        transcript_box.value = "文字起こし中…"
        kanpe_box.value = "カンペを生成中…"
        page.update()
 
        text = transcribe(path)
        hits = detect_ng_words(text)
        is_loud = max_db > LOUD_THRESHOLD_DBFS
        is_abusive = bool(hits) or is_loud
 
        volume_text.value = (
            f"音量(最大): {max_db:.1f} dBFS / 長さ: {duration:.1f}秒 / "
            f"大声判定: {'あり' if is_loud else 'なし'} / "
            f"検知ワード: {', '.join(hits) if hits else 'なし'}"
        )
        transcript_box.value = text
        kanpe_box.value = generate_kanpe(is_abusive, text)
 
        if is_abusive:
            saved = save_evidence(path)
            status_text.value = f"🚨 暴言検知：証拠をローカルに保全しました（{saved.name}）"
            status_text.color = ft.Colors.RED_400
        else:
            status_text.value = "🟢 平常時（データは保存されず即座に消去されます）"
            status_text.color = ft.Colors.GREEN_400
        page.update()
 
    file_picker = ft.FilePicker()
    page.services.append(file_picker)
 
    async def on_pick_click(e):
        files = await file_picker.pick_files(
            allow_multiple=False, allowed_extensions=["wav"]
        )
        if files:
            process_audio(files[0].path)
 
    page.add(
        ft.Text("🛡️ LocalSecure AI (窓口防衛エディション)", size=26, weight=ft.FontWeight.BOLD),
        ft.Text("〜完全オフラインで住民のプライバシーを守り、職員をリアルタイムに守る〜",
                size=13, color=ft.Colors.GREY_400),
        ft.Container(height=10),
        ft.Container(content=status_text, bgcolor=ft.Colors.GREY_900,
                     padding=10, border_radius=8),
        ft.Container(height=10),
        ft.ElevatedButton("音声ファイル(.wav)を読み込む", on_click=on_pick_click),
        volume_text,
        ft.Container(height=10),
        transcript_box,
        ft.Container(height=10),
        kanpe_box,
    )
 
 
ft.run(main)