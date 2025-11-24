# AnchorPro
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import os
import traceback

from anchor_pro.excel_interface import ExcelTablesImporter, ProjectController

import cProfile
import pstats
import io
import multiprocessing
import time


# # plots.matplotlib.use('TkAgg')
# import math
# import subprocess

# Inputs from powershell
# excel_path = sys.argv[1]
# output_dir = sys.argv[2]

# Manual Inputs for testing
# excel_path = r"C:\Users\djmiller\OneDrive - Degenkolb Engineers\AnchorPro\Reference Files\AnchorPro Input.xlsm"
# output_dir = r'C:\Users\djmiller\Desktop\AnchorPro Local'

# [NPC Testing]
# excel_path = r"C:\Users\djmiller\OneDrive - Degenkolb Engineers\AnchorPro\Reference Files\AnchorPro Input - NPC Beta Testing.xlsm"
# output_dir = r"C:\Users\djmiller\Desktop\AnchorPro Local\NPC Demo"

# User Testing
excel_path = r"C:\Users\djmiller\Desktop\AnchorPro Local\Nick Testing\251023_SMC Library_AnchorPro Input - v4.2.0.xlsm"
output_dir = r"C:\Users\djmiller\Desktop\AnchorPro Local\Nick Testing"


# DVC Machine Shop
# excel_path = r'\\degenkolb.corp\degdata\Projects\Oak\Project.C03\498\C3498013.00\Calcs\Active\Workshts\CD-Calculations\Equipment-Anchorage\AnchorPro Input - v4.2.0.xlsm'
# output_dir = r'C:\Users\djmiller\Desktop\AnchorPro Local\DVC'

# HNI OPM
# excel_path = r"C:\Users\djmiller\Desktop\AnchorPro Local\HNI\AnchorPro Input - HNI Terrace.xlsm"
# output_dir = r"C:\Users\djmiller\Desktop\AnchorPro Local\HNI"

# excel_path = r"C:\Users\djmiller\Desktop\AnchorPro Local\HNI\AnchorPro Input - HNI Terrace.xlsm"
# output_dir = r"C:\Users\djmiller\Desktop\AnchorPro Local\HNI\Terrace"

# Code Profiling Option
profile_code = False

def run_program(excel_path,output_dir):
    controller = ProjectController(excel_path, output_dir)
    controller.run_analysis()
    if controller.excel_tables.project_info['report_mode'] == 'Summary and Report':
        controller.create_report()
    return controller

# def profile_run_program():
#     import cProfile
#     import pstats
#     cProfile.runctx("run_program()", globals(), locals(), filename='profile_output.prof')
#     stats = pstats.Stats('profile_output.prof')
#     stats.sort_stats('cumulative')  # Sort by cumulative time
#     stats.reverse_order()
#     stats.print_stats()


class MultiStream:
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
    start_time = time.time()
    log_file_path = os.path.join(output_dir, 'LOG.txt')

    with open(log_file_path, 'w') as log_file:

        # Redirect stdout to both console and log file
        sys.stdout = MultiStream(sys.stdout, log_file)
        sys.stderr = MultiStream(sys.stderr, log_file)  # Redirect stderr as well
        print('#' * 40)
        print('Welcome to AnchorPro!')
        print('#' * 40)
        try:
            controller = run_program(excel_path, output_dir)

        except Exception as e:
            controller = None
            print("An error occurred:")
            traceback.print_exc()  # This will print the full traceback to both console and log file
        finally:
            print(f'Done. Elapsed time: {time.time() - start_time:.4f}')
            # Reset stdout and stderr to their original values

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        return controller

if __name__ == '__main__':
    # print('Initializing program...')

    multiprocessing.freeze_support()
    if profile_code:
        profiler = cProfile.Profile()
        profiler.enable()

        controller = main()

        profiler.disable()

        # Capture profile stats into a string buffer
        output = io.StringIO()
        stats = pstats.Stats(profiler, stream=output)
        stats.strip_dirs().sort_stats("cumulative").print_stats()

        # Reverse the output
        lines = output.getvalue().split("\n")
        for line in reversed(lines):
            print(line)

        # Save profile stats to a file
        profiler.dump_stats(os.path.join(output_dir, 'profile.txt'))
    else:
        controller = main()
