from anchor_pro.project_controller.project_controller import (
    ProjectController)
import time
import multiprocessing as mp


def run_program():
    excel_path = r"C:\Users\djmiller\OneDrive - Degenkolb Engineers\AnchorPro\Reference Files\AnchorPro Input.xlsm"
    output_dir = r'C:\Users\djmiller\Desktop\AnchorPro Local'

    # HNI OPM
    # excel_path = r"C:\Users\djmiller\Desktop\AnchorPro Local\HNI\AnchorPro Input - HNI Cabinets.xlsm"
    # output_dir = r"C:\Users\djmiller\Desktop\AnchorPro Local\HNI"

    # Instantiate Controller
    import_start = time.time()
    controller = ProjectController(excel_path, output_dir)

    # Run Analysis
    analysis_start = time.time()
    results = controller.run()

    # Print Results

    print("\n" + "=" * 60)
    print(f"Reporting Results")
    print("=" * 60)

    summary_start = time.time()
    controller.create_excel_summary_table()
    report_start = time.time()

    print("Printing report")
    controller.create_report()
    end_time=time.time()
    print("-"*60)
    print(f'Excel Import: {analysis_start - import_start:.4f} sec\n'
          f'Analysis: {summary_start - analysis_start:.4f} sec\n'
          f'Summary Table: {report_start - summary_start:.4f} sec\n'
          f'PDF Report: {end_time - report_start:.4f} sec\n'
          f'Total Time: {end_time-import_start:.2f} sec\n'
          f'Done.')
    print('-' * 60)

    # Optionally: summarize failures
    failed = [k for k, v in controller.model_records.items() if (v.model is None) or (len(v.analysis_runs) == 0)]
    if failed:
        print("\n" + "=" * 60)
        print("!" * 60)
        print("Models with errors:")
        for name in failed:
            print(f" - {name}")
        print("!" * 60)
        print("=" * 60)

    return controller
if __name__ == "__main__":
        mp.set_start_method("spawn", force=True)  # safest
        controller = run_program()


        # Test residual
        import numpy as np
        mr = controller.model_records['Wall Bracket 01']
        sol = mr.analysis_runs[0].solutions['LRFD']
        loads = mr.model.factored_loads.lrfd

        for i, t in enumerate(sol.theta_z):
            p = mr.model.get_load_vector(loads.Fh, loads.Fv, t)[:,0]
            u = sol.equilibrium_solutions[:,i]
            K = mr.model.update_stiffness_matrix(u)
            residual = np.dot(K, u) - p
            print(np.linalg.norm(residual))



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