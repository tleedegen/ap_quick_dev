
import sys
import os
import traceback
from anchor_pro.project_controller.project_controller import (
    ProjectController)
import time
import multiprocessing as mp


# Inputs from powershell
excel_path = sys.argv[1]
output_dir = sys.argv[2]

# Manual Inputs for testing
# excel_path = r"C:\Users\djmiller\OneDrive - Degenkolb Engineers\AnchorPro\Reference Files\AnchorPro Input.xlsm"
# output_dir = r'C:\Users\djmiller\Desktop\AnchorPro Local'

# HNI OPM
# excel_path = r"C:\Users\djmiller\Desktop\AnchorPro Local\HNI\AnchorPro Input - HNI Cabinets.xlsm"
# output_dir = r"C:\Users\djmiller\Desktop\AnchorPro Local\HNI"

# User Troubleshooting
# excel_path = r"C:\Users\djmiller\Desktop\AnchorPro Local\Rory Testing\AnchorPro Input - v4.2.11.xlsm"
# output_dir = r"C:\Users\djmiller\Desktop\AnchorPro Local\Rory Testing"

class MultiStream:
    """ A class to allow printing to console and a log file concurrently"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()  # Ensure immediate output

    def flush(self):
        for stream in self.streams:
            stream.flush()

def main():
    log_file_path = os.path.join(output_dir, 'LOG.txt')
    with open(log_file_path, 'w') as log_file:

        # Redirect stdout to both console and log file
        sys.stdout = MultiStream(sys.stdout, log_file)
        sys.stderr = MultiStream(sys.stderr, log_file)  # Redirect stderr as well
        print('#' * 60)
        print('Welcome to AnchorPro!')
        print('Version 4.2.10')
        print('#' * 60)


        try:
            controller = run_program()
        except Exception as e:
            controller = None
            print("An error occurred:")
            traceback.print_exc()  # This will print the full traceback to both console and log file

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        return controller


def run_program():
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
        # Print to LOG File
        print("\n" + "=" * 60)
        print("!" * 60)
        print("Models with errors:")
        for name in failed:
            print(f" - {name}")
        print("!" * 60)
        print("=" * 60)

        # Print Summary to auxiliary file for excel to read and display
        error_summary = os.path.join(output_dir, 'error_summary.txt')
        if failed:
            with open(error_summary, 'w') as es:
                es.write("Models with errors:\n")
                for name in failed:
                    es.write(f" - {name}\n")
        else:
            # Remove old summary if it exists so Excel doesn't show stale issues
            if os.path.exists(error_summary):
                os.unlink(error_summary)


    return controller


if __name__ == "__main__":
        mp.freeze_support()
        mp.set_start_method("spawn", force=True)  # safest
        controller = main()