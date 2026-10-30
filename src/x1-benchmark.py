"""
LocalSecure AI 検知テスト（測定ツール）

アプリ本体と同じ処理（音量解析 → 文字起こし → 暴言判定 → カンペ生成）を、
音声ファイルに対してまとめて実行し、検知率・誤検知率・処理時間を出します。
アプリ本体のファイルは書き換えません（同じフォルダに置いて使います）。

【使い方】
  1) 台本を読んで録音する（40件の録音データを作る）
       python benchmark.py record benchmark_sentences.txt
     → benchmark_data/abusive/（暴言）と benchmark_data/normal/（普通）にwavが保存されます

  2) 測定する
       python benchmark.py run benchmark_data
     → 画面に結果を表示し、benchmark_data/ に次の2ファイルを書き出します
        benchmark_summary.txt … 検知率・誤検知率・処理時間のまとめ（発表用）
        benchmark_result.csv  … 1件ごとの詳細（見逃し・誤検知の原因調べ用）

【オプション】
  --scene シーン名   測定に使うシーン（初期値: アプリの初期シーン）
  --kanpe            カンペ生成まで含めた時間も測る（Ollamaを起動しておく）
  --app ファイル名   読み込むアプリのファイル（初期値: 同じフォルダで最も新しいアプリ）

録音済みのwavがあれば、record を使わず、abusive/ と normal/ に入れても測定できます。
"""

import argparse
import csv
import importlib.util
import platform
import queue
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
LABEL_JP = {"abusive": "暴言", "normal": "普通"}


# ---------------------------------------------------------------- アプリの読み込み
def find_app(arg):
    if arg:
        p = Path(arg)
        if not p.exists() and (HERE / arg).exists():
            p = HERE / arg
        if not p.exists():
            sys.exit(f"アプリのファイルが見つかりません: {arg}")
        return p
    me = Path(__file__).resolve().name
    cands = [f for f in HERE.glob("*.py")
             if f.name != me and "ft.run(main)" in f.read_text(encoding="utf-8", errors="ignore")]
    if not cands:
        sys.exit("アプリのファイルが見つかりません。benchmark.py をアプリと同じフォルダに置くか、--app で指定してください。")
    return max(cands, key=lambda f: f.stat().st_mtime)


def load_app(path):
    """アプリのファイルを読み込む（画面は起動しない）"""
    try:
        import flet as ft
        ft.run = lambda *a, **k: None   # 末尾の ft.run(main) を何もしない関数に差し替える
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location("app_under_test", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_scene(mod, name):
    name = name or mod.DEFAULT_SCENE
    if name not in mod.SCENES:
        sys.exit(f"シーン「{name}」がありません。選べるシーン: {', '.join(mod.SCENES)}")
    return name, mod.SCENES[name]


# ---------------------------------------------------------------- 録音モード
def parse_sentences(path):
    items = []
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split("|", 2)]
        if len(parts) != 3 or parts[0] not in ("暴言", "普通"):
            sys.exit(f"{path} の{ln}行目の形式が違います（「暴言|大声|文」または「普通|普通|文」）: {line}")
        items.append(("abusive" if parts[0] == "暴言" else "normal", parts[1], parts[2]))
    return items


def record_once(mod, sd):
    """Enterで録音開始、もう一度Enterで停止。(音声, サンプルレート) を返す"""
    block_q = queue.Queue()
    stream = mod.open_mic_stream(sd, block_q)
    sr = int(stream.samplerate)
    stream.start()
    try:
        input("  ● 録音中…  話し終わったら Enter で停止 > ")
    finally:
        stream.stop()
        stream.close()
    blocks = []
    while not block_q.empty():
        blocks.append(block_q.get())
    if not blocks:
        return np.zeros(0, dtype="float32"), sr
    audio = np.concatenate(blocks)
    audio = audio[: max(len(audio) - int(sr * 0.15), 0)]   # 停止のEnterキーの音を落とす
    return audio, sr


def cmd_record(args):
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        sys.exit("録音には sounddevice が必要です: pip install sounddevice")
    mod = load_app(find_app(args.app))
    scene_name, scene = get_scene(mod, args.scene)
    items = parse_sentences(args.sentences)
    out = Path(args.out)
    (out / "abusive").mkdir(parents=True, exist_ok=True)
    (out / "normal").mkdir(parents=True, exist_ok=True)
    script_csv = out / "script.csv"
    if not script_csv.exists():
        with open(script_csv, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(["file", "label", "voice", "text"])

    print(f"台本 {len(items)}件。シーン「{scene_name}」の大声基準は {scene['loud_dbfs']:.0f} dBFS です。")
    print("各文を、指定された声の大きさで読み上げてください。録音済みの文は飛ばします（途中再開できます）。\n")

    for i, (label, voice, text) in enumerate(items, 1):
        fname = f"{'a' if label == 'abusive' else 'n'}{i:03d}.wav"
        path = out / label / fname
        if path.exists():
            continue
        while True:
            print(f"[{i}/{len(items)}] 区分: {LABEL_JP[label]} / 声: {voice}")
            print(f"  「{text}」")
            input("  Enter で録音開始 > ")
            audio, sr = record_once(mod, sd)
            if len(audio) < sr * 0.3:
                print("  録音が短すぎます。もう一度。\n")
                continue
            sf.write(str(path), audio, sr, subtype="PCM_16")
            max_db, dur = mod.analyze_volume(str(path))
            print(f"  長さ {dur:.1f}秒 / 最大音量 {max_db:.1f} dBFS（大声基準 {scene['loud_dbfs']:.0f}）")
            ans = input("  Enter=保存 / r=録り直し / s=この文を飛ばす / q=終了 > ").strip().lower()
            if ans == "r":
                path.unlink(missing_ok=True)
                print()
                continue
            if ans == "s":
                path.unlink(missing_ok=True)
                print()
                break
            if ans == "q":
                path.unlink(missing_ok=True)   # 保存を確定していない録音は残さない
                print("中断しました。もう一度同じコマンドで続きから録音できます。")
                return
            with open(script_csv, "a", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow([f"{label}/{fname}", label, voice, text])
            print("  保存しました。\n")
            break
    print("録音が終わりました。次は測定です:  python benchmark.py run", out)


# ---------------------------------------------------------------- 測定モード
def read_script(data):
    info = {}
    p = data / "script.csv"
    if p.exists():
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                info[r["file"].replace("\\", "/")] = (r.get("voice", ""), r.get("text", ""))
    return info


def pct(n, d):
    return f"{n / d * 100:.1f}%" if d else "-"


def metrics(rows, pred_fn):
    pos = [r for r in rows if r["label"] == "abusive"]
    neg = [r for r in rows if r["label"] == "normal"]
    tp = sum(1 for r in pos if pred_fn(r))
    fp = sum(1 for r in neg if pred_fn(r))
    return {"pos": len(pos), "neg": len(neg), "tp": tp, "fn": len(pos) - tp,
            "fp": fp, "tn": len(neg) - fp}


def time_line(label, values):
    if not values:
        return f"  {label}: -"
    return (f"  {label}: 平均 {statistics.mean(values):.1f}秒 / 中央値 {statistics.median(values):.1f}秒 / "
            f"最大 {max(values):.1f}秒")


def build_summary(mod, scene_name, scene, rows, errors, with_kanpe):
    L = []
    L.append("LocalSecure AI 検知テスト結果")
    L.append(f"シーン: {scene_name}（大声基準 {scene['loud_dbfs']:.0f} dBFS / 検知ワード: {', '.join(scene['ng_words'])}）")
    L.append(f"文字起こし: Whisper {mod.WHISPER_MODEL} / カンペ: {mod.OLLAMA_MODEL} / {platform.system()} {platform.processor() or platform.machine()}")
    L.append("")
    m = metrics(rows, lambda r: r["pred"])
    L.append(f"【検知の結果】 暴言 {m['pos']}件 / 普通 {m['neg']}件")
    L.append(f"  暴言 → 検知できた {m['tp']} / 見逃した {m['fn']}    検知率 {pct(m['tp'], m['pos'])}")
    L.append(f"  普通 → 正しく通した {m['tn']} / 誤検知 {m['fp']}    誤検知率 {pct(m['fp'], m['neg'])}")
    L.append(f"  全体の正解率 {pct(m['tp'] + m['tn'], m['pos'] + m['neg'])}")
    L.append("")
    L.append("【判定方法ごとの比較】")
    for name, fn in [("ワードのみ", lambda r: bool(r["hits"])),
                     ("大声のみ", lambda r: r["loud"]),
                     ("ワード+大声（現行）", lambda r: r["pred"])]:
        x = metrics(rows, fn)
        L.append(f"  {name}: 検知率 {pct(x['tp'], x['pos'])} / 誤検知率 {pct(x['fp'], x['neg'])}")
    L.append("")
    L.append("【処理時間（1件あたり）】")
    L.append(time_line("文字起こしと判定まで", [r["t_det"] for r in rows]))
    ks = [r["t_total"] for r in rows if r["t_total"] is not None]
    if with_kanpe:
        L.append(time_line("カンペ生成まで（合計）", ks))
    L.append(f"  音声の平均の長さ: {statistics.mean(r['duration'] for r in rows):.1f}秒"
             if rows else "")
    L.append("  ※話し終わりの判定（無音0.8秒）の待ち時間は含みません。")
    L.append("")
    missed = [r for r in rows if r["label"] == "abusive" and not r["pred"]]
    falsep = [r for r in rows if r["label"] == "normal" and r["pred"]]
    L.append(f"【見逃した暴言】 {len(missed)}件")
    for r in missed:
        L.append(f"  {r['file']}  最大音量 {r['max_db']:.1f}dBFS / 認識結果: {r['text']}")
    L.append(f"【誤検知した普通の発言】 {len(falsep)}件")
    for r in falsep:
        why = []
        if r["hits"]:
            why.append("ワード: " + ",".join(r["hits"]))
        if r["loud"]:
            why.append(f"大声 {r['max_db']:.1f}dBFS")
        L.append(f"  {r['file']}  {' / '.join(why)} / 認識結果: {r['text']}")
    if errors:
        L.append("")
        L.append(f"【測定できなかったファイル】 {len(errors)}件（上の数字には含みません）")
        for r in errors:
            L.append(f"  {r['file']}  {r['note']}")
    L.append("")
    L.append("※自分たちで用意した少数のデータでの検証です。話者・言い回しの偏りがあるため、参考値として扱ってください。")
    return "\n".join(L)


def cmd_run(args):
    data = Path(args.data)
    files = []
    for label in ("abusive", "normal"):
        for p in sorted((data / label).glob("*.wav")):
            files.append((label, p))
    if not files:
        sys.exit(f"{data}/abusive と {data}/normal に wav がありません。先に record で録音するか、wavを入れてください。")

    app_path = find_app(args.app)
    print(f"アプリ: {app_path.name}")
    mod = load_app(app_path)
    scene_name, scene = get_scene(mod, args.scene)
    script = read_script(data)

    print("Whisperを準備しています…（初回は時間がかかります）")
    try:
        mod.get_model()
    except Exception as ex:
        sys.exit(f"Whisperを読み込めませんでした（pip install faster-whisper を確認）: {ex}")
    with_kanpe = args.kanpe
    if with_kanpe:
        ok_w, ok_o = mod.warm_up()
        if not ok_o:
            print("Ollamaに接続できないため、カンペ生成の時間は測りません（Ollamaを起動してから再実行してください）。")
            with_kanpe = False
    mod.transcribe(str(files[0][1]))   # 1回目は読み込みで遅いので、測定に入れず捨てる

    rows, errors = [], []
    for i, (label, p) in enumerate(files, 1):
        rel = f"{label}/{p.name}"
        voice, said = script.get(rel, ("", ""))
        row = {"file": rel, "label": label, "voice": voice, "script": said,
               "hits": [], "loud": False, "pred": False, "max_db": 0.0, "duration": 0.0,
               "text": "", "t_det": 0.0, "t_total": None, "note": ""}
        try:
            t0 = time.perf_counter()
            max_db, dur = mod.analyze_volume(str(p))
            text = mod.transcribe(str(p))
            # アプリ本体（process_audio_core）と同じ判定
            hits = mod.detect_ng_words(text, scene["ng_words"])
            loud = max_db > scene["loud_dbfs"]
            pred = bool(hits) or loud
            t_det = time.perf_counter() - t0
            if text.startswith("【仮】") or text.startswith("（文字起こしに失敗"):
                row["note"] = "文字起こしに失敗（仮の文章が返ったため除外）"
                row["text"] = text
                errors.append(row)
                print(f"[{i}/{len(files)}] {rel}  文字起こし失敗")
            else:
                row.update(hits=hits, loud=loud, pred=pred, max_db=max_db, duration=dur,
                           text=text, t_det=t_det)
                if with_kanpe:
                    mod.generate_kanpe(pred, text, scene)
                    row["t_total"] = time.perf_counter() - t0
                rows.append(row)
                print(f"[{i}/{len(files)}] {rel}  {'検知' if pred else '通過'}  {t_det:.1f}秒  {text[:30]}")
        except Exception as ex:
            row["note"] = f"処理に失敗: {ex}"
            errors.append(row)
            print(f"[{i}/{len(files)}] {rel}  エラー: {ex}")

    if not rows:
        sys.exit("測定できたファイルがありません。Whisperの動作を確認してください。")

    summary = build_summary(mod, scene_name, scene, rows, errors, with_kanpe)
    (data / "benchmark_summary.txt").write_text(summary, encoding="utf-8")

    with open(data / "benchmark_result.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ファイル", "正解", "判定", "結果", "検知ワード", "最大音量(dBFS)", "大声判定",
                    "長さ(秒)", "認識結果", "台本", "声の指定", "文字起こしまで(秒)", "カンペまで(秒)", "備考"])
        for r in rows + errors:
            if r["note"]:
                result = "測定不可"
            elif r["label"] == "abusive":
                result = "検知できた" if r["pred"] else "見逃し"
            else:
                result = "誤検知" if r["pred"] else "正しく通過"
            w.writerow([r["file"], LABEL_JP[r["label"]], "暴言" if r["pred"] else "普通", result,
                        " ".join(r["hits"]), f"{r['max_db']:.1f}", "あり" if r["loud"] else "なし",
                        f"{r['duration']:.1f}", r["text"], r["script"], r["voice"],
                        f"{r['t_det']:.2f}", "" if r["t_total"] is None else f"{r['t_total']:.2f}", r["note"]])

    print("\n" + "=" * 60)
    print(summary)
    print("=" * 60)
    print(f"保存しました: {data / 'benchmark_summary.txt'} / {data / 'benchmark_result.csv'}")


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser(description="LocalSecure AI 検知テスト（測定ツール）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="台本を読み上げて録音する")
    r.add_argument("sentences", help="台本ファイル（benchmark_sentences.txt）")
    r.add_argument("--out", default="benchmark_data", help="保存先フォルダ（初期値: benchmark_data）")
    r.add_argument("--scene", help="大声基準の表示に使うシーン")
    r.add_argument("--app", help="アプリのファイル名")
    r.set_defaults(func=cmd_record)

    q = sub.add_parser("run", help="録音データを流して、検知率・誤検知率・処理時間を出す")
    q.add_argument("data", help="録音データのフォルダ（abusive/ と normal/ を含む）")
    q.add_argument("--scene", help="測定に使うシーン")
    q.add_argument("--kanpe", action="store_true", help="カンペ生成まで含めた時間も測る")
    q.add_argument("--app", help="アプリのファイル名")
    q.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
