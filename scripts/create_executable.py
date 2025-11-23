"""
Note the following hacky solution was required to get vtk to work with miniforge:
    https://stackoverflow.com/questions/21835851/python-executable-with-vtk-pyinstaller-py2exe-cx-freeze-etc
"""
import os
import PyInstaller.__main__

# Set environment variables
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'


# Path to additional project resources

conda_env_path = r'C:\Users\djmiller\Miniforge3\envs\env_ap'
graphics_path = '..\\graphics'

# PyInstaller.__main__.run([
#     '..\\anchor_pro\\main.py',  # Path to the main Python script you want to convert
#     '--onefile',  # Create a one-file bundled executable
#     # '--windowed',  # Use this option for GUI applications (remove it for console applications)
#     '--hidden-import=openpyxl.cell._writer',
#     '--hidden-import=matplotlib.backends.backend_pdf',
#     # '--exclude-module=traitsui main.py',
#     '--icon=..\\graphics\\DegPyramid.ico',
#     f'--add-data={graphics_path};graphics',
#     '--name=AnchorPro - v2.2.13'
#     # Add any additional options you need here
# ])

# PyInstaller.__main__.run([
#     '..\\anchor_pro\\main.py',

#     # Create a single-file executable (you can switch to --onedir for debugging)
#     '--onefile',

#     # Main app configuration
#     '--name=AnchorPro - v2.2.13',
#     '--icon=..\\graphics\\DegPyramid.ico',

#     # Include VTK hidden imports
#     '--hidden-import=vtkmodules.vtkCommonCore',
#     '--hidden-import=vtkmodules.vtkCommonMath',
#     '--hidden-import=vtkmodules.vtkCommonDataModel',
#     '--hidden-import=vtkmodules.vtkCommonExecutionModel',
#     '--hidden-import=vtkmodules.vtkCommonTransforms',
#     '--hidden-import=vtkmodules.vtkCommonMisc',
#     '--hidden-import=vtkmodules.vtkFiltersCore',
#     '--hidden-import=vtkmodules.vtkFiltersGeneral',
#     '--hidden-import=vtkmodules.vtkFiltersGeometry',
#     '--hidden-import=vtkmodules.vtkFiltersSources',
#     '--hidden-import=vtkmodules.vtkIOCore',
#     '--hidden-import=vtkmodules.vtkIOXML',
#     '--hidden-import=vtkmodules.vtkRenderingCore',
#     '--hidden-import=vtkmodules.vtkRenderingOpenGL2',
#     '--hidden-import=vtkmodules.vtkRenderingFreeType',
#     '--hidden-import=vtkmodules.vtkRenderingContext2D',
#     '--hidden-import=vtkmodules.vtkRenderingSceneGraph',
#     '--hidden-import=vtkmodules.vtkRenderingVtkJS',
#     '--hidden-import=vtkmodules.vtkRenderingAnnotation',
#     '--hidden-import=vtkmodules.vtkInteractionStyle',
#     '--hidden-import=vtkmodules.vtkInteractionWidgets',
#     '--hidden-import=vtkmodules.vtkWebCore',
#     '--hidden-import=vtkmodules.vtkWebGLExporter',
#     '--hidden-import=vtkmodules.vtkIOExport',
#     '--hidden-import=vtkmodules.vtkIOExportGL2PS',
#     '--hidden-import=vtkmodules.vtkViewsCore',
#     '--hidden-import=vtkmodules.vtkRenderingQt',

#     # Include additional project-specific hidden imports
#     '--hidden-import=openpyxl.cell._writer',
#     '--hidden-import=matplotlib.backends.backend_pdf',

#     # Include your app resources (e.g., icon and graphics)
#     f'--add-data={graphics_path};graphics',

#     # Bundle VTK binaries
#     f'--add-binary={vtk_bin_path}\\vtk*.dll;.',           # DLLs to root
#     f'--add-binary={vtk_pyd_path}\\*.pyd;vtkmodules'      # .pyds to vtkmodules
# ])




PyInstaller.__main__.run([
    'AnchorPro.spec'
])