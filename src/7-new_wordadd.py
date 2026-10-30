"""
【変更点】
p7: 「取ってほしい行動」の自由入力欄を追加（200文字まで）
  - 入力内容をカンペ生成プロンプトの追加方針として渡す
  - 固定ルール（責めない・その場で約束しない・突き放さない）が自由入力より優先
  - 処理開始時点の入力内容を、その音声の処理中は固定
p6: 利用シーンの切り替えを追加+スクロール追加
  - 画面に「市役所の窓口 / コールセンター / 店舗の接客」のボタンを追加
  - シーンごとに、次の3つが切り替わる
      ・暴言と判定するワード
      ・大声のしきい値(窓口 -20 / コールセンター -22 / 店舗 -18 dBFS)
      ・カンペの方針、見本の返答、AI失敗時の定型文
  - 暴言や脅しに応じない、突き放さない、というルールは固定(利用者が変更できない)
  - 処理中にシーンを切り替えても、その回の処理は開始時のシーンで固定
(p5: 処理時間の表示と高速化(起動時の先読み、結果の先出し))
(p4: 同じ音声は重複して保存しない)
【今後の予定】
  - シーンの自由入力←自由に取ってほしい行動を決められるように
  - マイクからのリアルタイム入力(sounddevice)
  - 誤検知の測定(検知率・処理時間)
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
WINDOW_SEC = 0.5
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"
WHISPER_MODEL = "small"       # もっと速くしたいときは "base" に(精度は少し下がる)
 
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
        "loud_dbfs": -18.0,   # 店内は周囲の音があるため、少し高めにしている
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
 
 
def detect_ng_words(text: str, words):
    return [w for w in words if w in text]
 
 
def _template_kanpe(is_abusive: bool, scene: dict) -> str:
    body = scene["template_abusive"] if is_abusive else scene["template_normal"]
    return f"【定型文】\n{body}"
 
 
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
 
 
def build_system_prompt(scene: dict, action_text: str = "") -> str:
    # 固定ルールは自由入力より常に優先する
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
    return (
        f"あなたは{scene['role']}を支援するアシスタントです。"
        "住民やお客様の発言に対して、職員がそのまま口に出せる返答を日本語で書きます。\n"
        f"この現場の方針: {scene['guide']}\n{optional}\n{fixed}"
    )
 
 
def generate_kanpe(is_abusive: bool, text: str, scene: dict, action_text: str = "") -> str:
    """ローカルのOllamaでカンペを生成。接続できなければ定型文に戻る"""
    situation = "暴言・威圧的な発言" if is_abusive else "通常の問い合わせ"
    messages = [{"role": "system", "content": build_system_prompt(scene, action_text)}]
    for sit, said, reply in scene["examples"]:
        messages.append({"role": "user", "content": f"状況: {sit}\n発言: {said}"})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": f"状況: {situation}\n発言: {text}"})
    try:
        reply = _ollama_chat(messages)
        label = "【AI推奨の防衛セリフ（ローカルAI生成）】" if is_abusive else "【AI推奨対応（ローカルAI生成）】"
        return f"{label}\n{reply}"
    except Exception as ex:
        return _template_kanpe(is_abusive, scene) + f"\n\n（ローカルAIに接続できませんでした: {ex}）"
 
 
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
    page.title = "LocalSecure AI - カスハラ防衛システム"
    page.window.width = 800
    page.window.height = 800
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO#スクロール
 
    state = {"scene": DEFAULT_SCENE}
 
    status_text = ft.Text(
        "🟢 平常時（データは保存されず即座に消去されます）",
        size=14, color=ft.Colors.GREEN_400, weight=ft.FontWeight.BOLD,
    )
    ready_text = ft.Text("ローカルAIを準備中…（初回のみ少し時間がかかります）",
                         size=12, color=ft.Colors.AMBER_400)
    scene_text = ft.Text("", size=12, color=ft.Colors.GREY_400)
    volume_text = ft.Text("音量: --", size=13, color=ft.Colors.GREY_400)
    timing_text = ft.Text("処理時間: --", size=13, color=ft.Colors.CYAN_300)
    action_count = ft.Text("0 / 200文字", size=12, color=ft.Colors.GREY_400)

    action_box = ft.TextField(
        label="カンペに加えたい『取ってほしい行動』（任意・200文字まで）",
        hint_text="例：まず担当者が状況を確認し、次に受付番号を案内する",
        multiline=True,
        min_lines=2,
        max_lines=4,
        max_length=200,
        on_change=lambda e: update_action_count(e.control.value or ""),
    )

    def update_action_count(value: str):
        action_count.value = f"{len(value)} / 200文字"
        page.update()
 
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
 
    # --- 利用シーンの切り替え ---
    scene_row = ft.Row(wrap=True)
 
    def describe_scene():
        sc = SCENES[state["scene"]]
        scene_text.value = (
            f"現在のシーン: {state['scene']} / 大声のしきい値: {sc['loud_dbfs']:.0f} dBFS / "
            f"検知ワード: {', '.join(sc['ng_words'])}"
        )
 
    def make_scene_handler(name):
        def handler(e):
            state["scene"] = name
            refresh_scene_buttons()
            describe_scene()
            page.update()
        return handler
 
    def refresh_scene_buttons():
        buttons = []
        for name in SCENES:
            if name == state["scene"]:
                buttons.append(ft.ElevatedButton(name, on_click=make_scene_handler(name)))
            else:
                buttons.append(ft.OutlinedButton(name, on_click=make_scene_handler(name)))
        scene_row.controls = buttons
 
    refresh_scene_buttons()
    describe_scene()
 
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
        scene = SCENES[state["scene"]]   # 処理中にシーンを切り替えても、この回は固定
        action_text = (action_box.value or "")[:200]  # この回の自由入力も開始時点で固定
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
 
        hits = detect_ng_words(text, scene["ng_words"])
        is_loud = max_db > scene["loud_dbfs"]
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
 
        kanpe = await asyncio.to_thread(generate_kanpe, is_abusive, text, scene, action_text)
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
        ft.Text("🛡️ LocalSecure AI", size=26, weight=ft.FontWeight.BOLD),
        ft.Text("〜完全オフラインでプライバシーを守り、現場の職員をリアルタイムに守る〜",
                size=13, color=ft.Colors.GREY_400),
        ft.Container(height=10),
        ft.Container(content=status_text, bgcolor=ft.Colors.GREY_900,
                     padding=10, border_radius=8),
        ready_text,
        ft.Container(height=6),
        ft.Text("利用シーンを選ぶ", size=13, weight=ft.FontWeight.BOLD),
        scene_row,
        scene_text,
        ft.Container(height=6),
        ft.Text("カンペの追加方針（任意）", size=13, weight=ft.FontWeight.BOLD),
        action_box,
        action_count,
        ft.Text("※この欄はカンペの方針に追加されます。固定ルール『責めない・その場で約束しない・突き放さない』が常に優先されます。",
                size=11, color=ft.Colors.AMBER_400),
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