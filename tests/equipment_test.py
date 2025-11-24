from anchor_pro.project_controller.project_controller import (
    ProjectController)
import time

excel_path = r"C:\Users\djmiller\OneDrive - Degenkolb Engineers\AnchorPro\Reference Files\AnchorPro Input.xlsm"
output_dir = r'C:\Users\djmiller\Desktop\AnchorPro Local'


# Instantiate Controller
import_start = time.time()
controller = ProjectController(excel_path, output_dir)

# Run Analysis
analysis_start = time.time()
results = controller.run()

# Print Results
summary_start = time.time()
controller.create_excel_summary_table()
report_start = time.time()

print("Printing report")
controller.create_report()
end_time=time.time()
print(f'Done. \n'
      f'Data Import: {analysis_start - import_start:.4f} sec\n'
      f'Analysis: {summary_start - analysis_start:.4f} sec\n'
      f'Summary Table: {report_start - summary_start:.4f} sec\n'
      f'PDF Report: {end_time - report_start:.4f} sec\n'
      f'Total Time: {end_time-import_start}')



print('Post-processing')

# import numpy as np
# from anchor_pro.model import FactorMethod
# record = controller.models['Base 02a']
# model = record.model
# run = record.analysis_runs[record.governing_run]
# anchor_fz = np.sum([plate.anchor_forces[:,2,:] for plate in run.results.base_plates if len(plate.anchor_forces)>0],axis=0)
# cz_fz = np.zeros(anchor_fz.shape[1])
# for plate in run.results.base_plates:
#     for i, cz in enumerate(plate.compression_zones):
#         cz_fz[i] += cz.fz.sum()
#
# sol_record = run.solutions[FactorMethod.lrfd_omega]
# sol = sol_record.equilibrium_solutions
# p = model.get_load_vector(*model.factored_loads.get(FactorMethod.lrfd_omega),sol_record.theta_z)
# res = model.equilibrium_residual(sol[:,0],p[:,0])