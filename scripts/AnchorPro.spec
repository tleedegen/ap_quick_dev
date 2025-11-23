# -*- mode: python ; coding: utf-8 -*-


import glob
import os

# Define VTK locations from your Miniforge environment
env_root = r'C:\\Users\\djmiller\\Miniforge3\\envs\\env_ap'
# vtk_bin = os.path.join(env_root, 'Library', 'bin')
# vtk_pyd = os.path.join(env_root, 'Lib', 'site-packages', 'vtkmodules')

# Collect all relevant VTK binaries and compiled Python extensions
# vtk_dlls = [(f, '.') for f in glob.glob(os.path.join(vtk_bin, 'vtk*.dll'))]
# vtk_pyds = [(f, 'vtkmodules') for f in glob.glob(os.path.join(vtk_pyd, '*.pyd'))]



a = Analysis(
    ['..\\anchor_pro\\main.py'],
    pathex=[],
   # binaries = vtk_dlls + vtk_pyds,
    datas=[('..\\graphics', 'graphics')],
    hiddenimports=[
        'openpyxl.cell._writer',
        'matplotlib.backends.backend_pdf'
    ],
    hookspath=['./my_hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

#One File Version
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AnchorPro - v4.2.3',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['..\\graphics\\DegPyramid.ico'],
)

# One Dir Version
#exe = EXE(
#    pyz,
#    a.scripts,
#    [],
#    exclude_binaries=True,
#    name='AnchorPro - v2.2.13',
#    debug=False,
#    bootloader_ignore_signals=False,
#    strip=False,
#    upx=True,
#    console=True,
#    icon=['..\\graphics\\DegPyramid.ico']
#)

#coll = COLLECT(
#    exe,
#    a.binaries,
#    a.zipfiles,
#    a.datas,
#    strip=False,
#    upx=True,
#    name='AnchorPro - v2.2.13'
#)