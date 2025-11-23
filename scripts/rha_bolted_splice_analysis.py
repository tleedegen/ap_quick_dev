from anchor_pro.bolted_column_splice import SpliceAnalyzer
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')


# Define File Paths
output_folder = r"C:\Users\djmiller\Desktop\EWT Output\SpliceResults"
wb_splice = r"\\degenkolb.corp\degdata\Projects\Sac\Project.B09\953\B9953031.03\Calcs\Active\Workshts\ColumnSpliceInfo_1.xlsx"
forces_path = r"\\degenkolb.corp\degdata\Projects\Sac\Project.B09\953\B9953031.03\Calcs\Active\Workshts\Splice Demands - Shortened"


# Initialize Paths to Splice Force Results
splice_forces_paths = {'GM01': os.path.join(forces_path, "Splice-Demand Trials - GM01.csv"),
                 'GM02': os.path.join(forces_path, "Splice-Demand Trials - GM02.csv"),
                 'GM03': os.path.join(forces_path, "Splice-Demand Trials - GM03.csv"),
                 'GM04': os.path.join(forces_path, "Splice-Demand Trials - GM04.csv"),
                 'GM05': os.path.join(forces_path, "Splice-Demand Trials - GM05.csv"),
                 'GM06': os.path.join(forces_path, "Splice-Demand Trials - GM06.csv"),
                 'GM07': os.path.join(forces_path, "Splice-Demand Trials - GM07.csv"),
                 'GM08': os.path.join(forces_path, "Splice-Demand Trials - GM08.csv"),
                 'GM09': os.path.join(forces_path, "Splice-Demand Trials - GM09.csv"),
                 'GM10': os.path.join(forces_path, "Splice-Demand Trials - GM10.csv"),
                 'GM11': os.path.join(forces_path, "Splice-Demand Trials - GM11.csv")}

# forces_path = r"\\degenkolb.corp\degdata\Projects\Sac\Project.B09\953\B9953031.03\Calcs\Active\Models\Perform3D\Process\v7\Splice 2E"
# splice_forces_paths = {'GM02': os.path.join(forces_path, 'Splice-GM02.csv')}


rn = 39 * 0.6

analyzer = SpliceAnalyzer(wb_splice, splice_forces_paths, output_folder, rn)

#
''' 
Analysis for a single Splice
    Instructions:
    Provide the splice number, ground motion label, and time step for a specific splice.
    The program will run the analysis and plot the resulting load state
'''
# splice, sol, p = analyzer.analyze_single_splice(180, gm_key='GM01', time_step=1577)
splice, sol, p = analyzer.analyze_single_splice(170, gm_key='GM02', time_step=266)
splice.plot_3d_wireframe(u=sol, p=p)
print(splice.resultant_direction_residual(sol))
'''
Analysis for all splices
    Instructions:
    Comment out the section above, and uncomment the section below.
'''
# analyzer.analyze_all_splices()  # Uncomment
#





