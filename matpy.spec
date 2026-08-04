# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files("tree_sitter_matlab", excludes=["**/_binding*"])
binaries = collect_dynamic_libs("tree_sitter_matlab")

a = Analysis(
    ["ui/minimal_window.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=["tree_sitter", "tree_sitter_matlab"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="matpy",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
