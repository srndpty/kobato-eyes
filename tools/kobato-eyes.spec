# tools/kobato-eyes.spec
# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_dynamic_libs

block_cipher = None

# プロジェクトのルートと src パス（src レイアウト前提）
ROOT = Path.cwd() # 常にプロジェクトルートでpyinstallerを実行しなければならない
# ROOT = Path(__file__).resolve().parents[1] # こちらは環境によってコケることあり。
SRC = ROOT / "src"

# --- PyQt6 は一括収集（プラグイン/翻訳/hiddenimports 含む）
pyqt_datas, pyqt_bins, pyqt_hidden = collect_all("PyQt6")

# --- hiddenimports
hiddenimports = []
hiddenimports += pyqt_hidden
hiddenimports += collect_submodules("onnxruntime")
hiddenimports += collect_submodules("hnswlib")
# （torch を使わないなら入れない方が軽い）

# --- binaries (DLL)
binaries = []
binaries += pyqt_bins
binaries += collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("hnswlib")

# --- datas（アプリ内アセットを同梱）
datas = []
datas += pyqt_datas
datas += [
    # 画像・アイコン類（アプリ側では utils.resources.resource_path 等で参照）
    (str(SRC / "ui" / "assets"), "ui/assets"),
]

a = Analysis(
    [str(SRC / "ui" / "app.py")],   # ← ここが重要：スクリプトの“実ファイルパス”
    pathex=[str(SRC)],              # ← src レイアウトなら pathex に src を通す
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # 配布に不要なら除外してサイズ/起動を削減
        "pytest", "hypothesis", "pip", "pip_audit", "pre_commit",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,                # onedir + zip化＝起動とI/Oのバランス良
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="kobato-eyes",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                      # ★ UPXは起動が遅くなるので無効
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                  # GUI アプリ
    icon=str(SRC / "ui" / "assets" / "icons" / "kobato_eye_right.ico"),  # EXEアイコン
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,                      # こちらも UPX 無効
    upx_exclude=[],
    name="kobato-eyes",            # onedir 配布
)
