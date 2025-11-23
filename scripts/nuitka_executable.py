import os
import subprocess
import time

tic = time.time()
# Set environment variables
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

# Paths
graphics_path = '../graphics'  # Adjusting for "scripts" subfolder
icon_path = '../graphics/DegPyramid.ico'  # Adjusting for "scripts" subfolder

# Build command for Nuitka
nuitka_command = [
    'python', '-m', 'nuitka',
    '--onefile',               # Create a single executable file
    '--standalone',            # Make the executable standalone (include all dependencies)
    '--enable-plugin=tk-inter', # Enable plugins for specific packages as needed
    '--include-data-dir=' + f'{graphics_path}=graphics',  # Include entire graphics folder
    '--windows-icon-from-ico=' + icon_path,  # Set the icon for the executable
    '--plugin-enable=pyqt5',
    '--output-filename=AnchorPro - 250130.exe',  # Set output file name
    '../anchor_pro/main.py'                # Path to the main script, adjusting for "scripts" subfolder
]

# Additional hidden imports for packages used in your project
hidden_imports = [
    '--include-module=openpyxl.cell._writer',
    '--include-module=matplotlib.backends.backend_pdf',
    '--include-data-file=C:\\Users\\djmiller\\.conda\\envs\\daea_env\\python38.dll=python38.dll'
]

# Combine all parts of the command
nuitka_command.extend(hidden_imports)

# Run the Nuitka command
subprocess.run(nuitka_command)

print('Done')
print(f'Time: {time.time() - tic}')