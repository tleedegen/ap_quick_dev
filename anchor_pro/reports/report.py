# Define the refactored code based on the given objectives
# from imghdr import test_gif

from pylatex import (Package, LineBreak, NewLine, MiniPage, Tabular, LongTable,
                     Document, Command, FlushRight, FlushLeft, LargeText,
                     MediumText,NewPage, Section, Subsection, Subsubsection,
                     Tabularx, Math, MultiColumn, Alignat, Enumerate, MultiRow)

from pylatex.utils import bold, NoEscape
from pylatex.base_classes import Environment

from typing import Union, List, Any, Tuple
import anchor_pro.model as m
import anchor_pro.elements.concrete_anchors as conc
import anchor_pro.elements.sms as sms
from anchor_pro.elements.sms import SMSCondition
from anchor_pro.model import WallMaterial
from anchor_pro.project_controller.project_classes import ModelRecord
from anchor_pro.utilities import get_governing_result

import anchor_pro.reports.plots as plots
import anchor_pro.config
from anchor_pro.ap_types import (FactorMethod, RESULTS_LIKE)

import numpy as np

import pandas as pd
import os

# import anchor_pro.calculator
# from io import BytesIO
# import tempfile
# import multiprocessing as mp
# import queue

import re

result_queue = None


class Flalign(Environment):
    """A class to wrap the LaTeX flalign environment."""
    omit_if_empty = True
    packages = [Package("amsmath")]
    _latex_name = 'flalign'

    def __init__(self, numbering=False, escape=False):
        self.numbering = numbering
        self.escape = escape
        if not numbering:
            self._star_latex_name = True
        super().__init__()


def subheader(container, text):
    container.append(NoEscape(r'\smallskip'))
    container.append(LineBreak())
    container.append(NoEscape(rf'\makebox[0pt][l]{{\textit{{\textbf{{{text}}}}}}}'))
    container.append(NewLine())



def subheader_nobreak(container, text):
    """Subheader to be used at the start of a section or minipage.
    If no previous content is present, the regular subheader will give an error:
    "No line here to end" due to the initial line break."""

    container.append(NoEscape(rf'\makebox[0pt][l]{{\textit{{\textbf{{{text}}}}}}}'))
    container.append(NewLine())
    # container.append(NoEscape(r'\smallskip'))


INVALID_WIN_CHARS = r'[<>:"/\\|?*\x00-\x1F]'

def make_figure_filename(sec_name, sub_sec, fig_name):
    # Create initial filename
    filename = f"{sec_name}, {sub_sec}, {fig_name}"

    # Remove invalid Windows filename characters
    clean_filename = re.sub(INVALID_WIN_CHARS, '', filename)

    # Collapse multiple whitespace characters into a single space
    clean_filename = re.sub(r'\s+', ' ', clean_filename)

    # Strip outer whitespace and trailing dots/spaces
    clean_filename = clean_filename.strip().rstrip('. ')

    return clean_filename


def make_figure(sec, width, file, title=None, pos='t',use_minipage=True):
    if use_minipage:
        with sec.create(MiniPage(width=f'{width:.2f}in', pos=pos, align='top')) as mini:
            mini.append(NoEscape(r'\centering'))
            if title:
                mini.append(NoEscape(rf'\textit{{\textbf{{{title}}}}}'))
                mini.append(NewLine())
                mini.append(NoEscape(r'\smallskip'))

            mini.append(NoEscape(rf'\includegraphics[width={width:.2f}in,valign=t]{{ {file} }}\\'))
    else:
        sec.append(NoEscape(rf'\includegraphics[width={width:.2f}in,valign=t]{{ {file} }}\\'))

def math_alignment_table(sec, math_lines: list[list] | list[tuple] | tuple[list] | tuple[tuple],
                         width='6.5in',pos='t') -> None:
    if not math_lines or not all(len(row) == len(math_lines[0]) for row in math_lines):
        raise ValueError("All rows must have the same number of columns and cannot be empty.")

    num_cols = len(math_lines[0])
    # All but last column are math-mode, last is text
    col_format = "".join([r'>{$}l<{$}' for _ in range(num_cols - 1)]) + "X"

    with sec.create(MiniPage(width=width, pos=pos)) as mini:
        with mini.create(Tabularx(NoEscape(col_format))) as table:
            for row in math_lines:
                table.add_row([NoEscape(text) for text in row])

def math_alignment_longtable(sec,
                                    math_lines: list[list] | list[tuple] | tuple[list] | tuple[tuple],
                                    width='6.5in',
                             omit_line_label:bool=False) -> None:
    if not math_lines or not all(len(row) == len(math_lines[0]) for row in math_lines):
        raise ValueError("All rows must have the same number of columns and cannot be empty.")

    num_cols = len(math_lines[0])

    # Estimate column width (in inches)
    if width.endswith('in'):
        try:
            total_width_in = float(width.rstrip('in'))
            col_width = total_width_in / num_cols
            col_width_str = f'{col_width:.3f}in'
        except ValueError:
            raise ValueError("Invalid width format. Use a numeric value ending in 'in', e.g., '6.5in'.")
    else:
        col_width_str = f"{1/num_cols:.3f}{width}"  # e.g., 0.25\textwidth

    # Build column spec: math-mode for all but last, right-aligned text last
    if omit_line_label:
        col_spec_parts = [
            r'>{$}p{' + col_width_str + r'}<{$}' for _ in range(num_cols)
        ]
    else:
        col_spec_parts = [
            r'>{$}p{' + col_width_str + r'}<{$}' for _ in range(num_cols - 1)
        ]

        col_spec_parts.append(r'>{\raggedleft\arraybackslash}p{' + col_width_str + r'}')

    col_format = "".join(col_spec_parts)

    # Start local group for zero tabcolsep
    sec.append(NoEscape(r'\begingroup'))
    sec.append(NoEscape(r'\setlength{\tabcolsep}{0pt}'))

    with sec.create(LongTable(NoEscape(col_format))) as table:
        for row in math_lines:
            table.add_row([NoEscape(text) for text in row])

    # End local group
    sec.append(NoEscape(r'\endgroup'))

def make_table(sec, title, header, units, data, alignment=None, col_formats=None,
               utilization_cols=[], utilization_limit=1,
               rows_to_highlight=None, add_index=True, width=r'\textwidth', pos='t', use_minipage=True,
               font_size='footnotesize', align='l'):

    """
    Create a table in a LaTeX document section with specified formatting and data.

    Parameters:
    - sec: LaTeX section object where the table will be added.
    - title: Title of the table.
    - header: List of column headers.
    - units: List of units corresponding to each column.
    - data: Data input (can be a Pandas DataFrame, NumPy array, or dictionary of lists).
    - alignment: Optional string specifying column alignments.
    - col_formats: Optional list of format strings for each column.
    - add_index: Boolean indicating whether to include a 1-based index column.
    """
    # Convert input data into a list of lists for uniform handling
    if isinstance(data, pd.DataFrame):
        data_values = data.to_numpy()
    elif isinstance(data, dict):
        data_values = list(zip(*data.values()))  # Convert dict of lists to list of tuples
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data_values = data[:, np.newaxis]  # Convert 1D array to 2D
        else:
            data_values = data
    elif isinstance(data, list):
        data_values = data
    else:
        raise TypeError("Unsupported data type. Expected DataFrame, NumPy array, or dictionary of lists.")

    num_cols = len(header)

    # Set default alignment and column formats if not provided
    if alignment is None:
        alignment = 'l' + 'c' * (num_cols - 1)

    # Add an index column if required
    if add_index:
        header = ['\#'] + header
        units = [''] + units
        alignment = 'c' + alignment
        col_formats = ['{:.0f}'] + col_formats
        utilization_cols = [idx + 1 for idx in utilization_cols]
        data_values = [(i + 1,) + tuple(row) for i, row in enumerate(data_values)]

    # Apply LaTeX-specific formatting for the header and units rows
    header[0] = NoEscape(r'\rowcolor{lightgray} ' + header[0])
    units[0] = NoEscape(r'\rowcolor{lightgray} ' + units[0])



    # Create the table in the LaTeX document
    if use_minipage:
        with sec.create(MiniPage(width=width, pos=pos, align=align)) as mini:
            if title:
                mini.append(NoEscape(rf'\textit{{\textbf{{{title}}}}}'))
                mini.append(NewLine())
                mini.append(NoEscape(r'\smallskip'))

            mini.append(NoEscape(f'\\begin{{{font_size}}}'))
            with mini.create(Tabular(alignment)) as table:
                populate_table(table, header, units, data_values,
                   col_formats, utilization_cols, utilization_limit, rows_to_highlight)
            mini.append(NoEscape(f'\\end{{{font_size}}}'))
    else:
        if title:
            sec.append(NoEscape(rf'\textit{{\textbf{{{title}}}}}'))
            sec.append(NewLine())
            # sec.append(NoEscape(r'\smallskip'))

        sec.append(NoEscape(f'\\begin{{{font_size}}}'))
        with sec.create(LongTable(alignment)) as table:
            populate_table(table, header, units,  data_values,
                   col_formats, utilization_cols, utilization_limit, rows_to_highlight)
        sec.append(NoEscape(f'\\end{{{font_size}}}'))

def populate_table(table, header, units, data_values,
                   col_formats, utilization_cols, utilization_limit, rows_to_highlight):
    table.add_hline()
    table.add_row(header)
    table.add_row(units)
    table.add_hline()

    # Add rows to the table
    for i, row in enumerate(data_values):
        formatted_row = [fmt.format(val) if val is not None else val for fmt, val in zip(col_formats, row)]
        # Apply text coloring if utilization_col is active
        for idx in utilization_cols:
            formatted_row[idx] = utilization_text_color(formatted_row[idx], row[idx], utilization_limit) \
                if row[idx] is not None else formatted_row[idx]

        if rows_to_highlight is None:
            highlight_set = set()
        elif isinstance(rows_to_highlight, (int, np.int_)):
            highlight_set = {rows_to_highlight}
        else:
            highlight_set = set(rows_to_highlight)

        highlight_this_row = i in highlight_set

        if highlight_this_row:
            formatted_row[0] = NoEscape(r'\rowcolor{yellow} ' + formatted_row[0])

        table.add_row(formatted_row)
        table.add_hline()


def utilization_text_color(cell, value, limit):
    if isinstance(value, str):
        if value == 'OK':
            color = 'Green'
        elif value == 'NG':
            color = 'red'
        else:
            raise Exception("Table column result must be either 'OK' or 'NG'")
    else:
        color = 'red' if value > limit else 'Green'
    return NoEscape(fr'\textcolor{{{color}}}{{ {cell} }}')


def insert_framed_pdf(section, pdf_path, subsection_title="Addendum"):
    """
    Inserts a framed PDF into the specified section, scaled to fit in the text area.

    Args:
    - section (pylatex.Section or pylatex.Subsection): The section object to insert the PDF into.
    - pdf_path (str): Path to the PDF file to be inserted.
    - subsection_title (str): Title of the subsection where the PDF will be inserted.
    """
    # Insert the LaTeX command to include the PDF, scaled to fit the text width and height
    section.append(NoEscape(
        r'\includepdf[pages=-, width=\textwidth, height=\textheight, keepaspectratio, frame=true, pagecommand={\pagestyle{StyleSectionSheet}}]{' + pdf_path.replace(
            '\\', '/') + '}'
    ))


class Report:
    def __init__(self, project_info, pool=None):
        self.project_info = project_info
        self.logo_path = os.path.join(anchor_pro.config.base_path, "graphics", "DegLogo.pdf").replace('\\', '/')
        self.doc = self.setup_document()
        self.pool = pool  # multiprocessing pool for figure generation

    def setup_document(self):
        geometry_options = {
            "margin": "1in",
            "top": "1in",
            "bottom": "1in",
        }
        document_options = ['fleqn']
        doc = Document(geometry_options=geometry_options, document_options=document_options)
        doc.preamble.append(NoEscape(r'\setlength{\headheight}{0.5in}'))

        doc.preamble.append(Package('times'))
        doc.preamble.append(Package('helvet'))
        doc.preamble.append(Package('mathptmx'))
        doc.preamble.append(Package('amsmath'))
        doc.packages.append(Package('adjustbox', options='export'))
        doc.packages.append(Package('xcolor', options=['table, dvipsnames']))
        doc.packages.append(Package('pgf'))
        doc.packages.append(Package('graphicx'))

        doc.packages.append(Package('hyperref', options=['hidelinks, bookmarksdepth=2, bookmarksnumbered']))

        doc.packages.append(Package('fancyhdr'))
        doc.preamble.append(NoEscape(r'\usepackage{sectsty}'))
        doc.packages.append(Package('pdfpages'))
        doc.preamble.append(NoEscape(r'\allsectionsfont{\sffamily}'))
        doc.preamble.append(NoEscape(r'\fancypagestyle{StyleSectionSheet}{'))
        doc.preamble.append(NoEscape(r'\fancyheadoffset{0in}'))
        doc.preamble.append(NoEscape(r'\fancyfootoffset{0in}'))
        doc.preamble.append(NoEscape(rf'\fancyhead[L]{{\includegraphics[height=.4in]{{{self.logo_path}}}}}'))
        doc.preamble.append(NoEscape(
            rf'\fancyhead[R]{{\sffamily {self.project_info["project_title"]} \\ {self.project_info["package_info1"]} }}'))
        doc.preamble.append(NoEscape(r'\fancyhead[C]{}'))
        doc.preamble.append(NoEscape(r'\fancyfoot[L,R]{}'))
        doc.preamble.append(NoEscape(r'\fancyfoot[C]{\thepage}'))
        doc.preamble.append(NoEscape(r'\renewcommand{\headrulewidth}{1pt}'))
        doc.preamble.append(NoEscape(r'\setlength{\headsep}{0.5in}'))
        doc.preamble.append(NoEscape(r'\renewcommand{\footrulewidth}{1pt}}'))
        doc.preamble.append(NoEscape(r'\renewcommand{\familydefault}{\sfdefault}'))
        doc.preamble.append(NoEscape(r'\setcounter{tocdepth}{2}'))
        doc.preamble.append(NoEscape(r'\everydisplay{\footnotesize}'))
        doc.packages.append(Package('array'))
        return doc

    def cover_page(self):
        doc = self.doc
        doc.append(NoEscape(r'\pdfbookmark[0]{Coverpage}{cover}'))
        self.add_logos_and_address(doc)
        self.add_project_info(doc)

    def add_logos_and_address(self, doc):
        with doc.create(MiniPage(width="4.5in")) as minipage:
            minipage.append(Command('includegraphics', options='width=3in', arguments=NoEscape(self.logo_path)))
        doc.append(NoEscape(r'\hfill'))
        with doc.create(MiniPage(width="1.95in")) as minipage:
            with minipage.create(FlushRight()) as right_aligned:
                right_aligned.append(bold('Degenkolb Engineers'))
                right_aligned.append(NewLine())
                right_aligned.append(self.project_info['address'])
                right_aligned.append(NewLine())
                right_aligned.append(self.project_info['address2'])
                right_aligned.append(NewLine())
                right_aligned.append(self.project_info['city'])
                right_aligned.append(NewLine())
                right_aligned.append(self.project_info['phone'])
                right_aligned.append(NewLine())

    def add_project_info(self, doc):
        with doc.create(FlushLeft()) as fl:
            # Project Information
            fl.append(NoEscape(r'\vspace{1in}'))
            fl.append(LargeText(bold(self.project_info['project_title'])))
            fl.append(NoEscape(r' \hrule \medskip '))
            if self.project_info['project_info1']:
                fl.append(MediumText(self.project_info['project_info1']))
                fl.append(NewLine())
            if self.project_info['project_info2']:
                fl.append(self.project_info['project_info2'])
                fl.append(NewLine())
            if self.project_info['project_info3']:
                fl.append(self.project_info['project_info3'])
                fl.append(NewLine())
            if self.project_info['project_info4']:
                fl.append(self.project_info['project_info4'])
            fl.append(NoEscape(r'\vfill'))

            # Package Info
            if self.project_info['package_info1']:
                fl.append(LargeText(self.project_info['package_info1']))
                fl.append(NewLine())
            if self.project_info['package_info2']:
                fl.append(self.project_info['package_info2'])
                fl.append(NewLine())
            if self.project_info['package_info3']:
                fl.append(self.project_info['package_info3'])
                fl.append(NewLine())
            if self.project_info['package_info4']:
                fl.append(self.project_info['package_info4'])
                fl.append(NewLine())
            fl.append(NoEscape(r'\vfill'))

            # Job Numbers
            if self.project_info['job_number']:
                fl.append(NoEscape(rf'Degenkolb Job Number: {self.project_info["job_number"]}'))
                fl.append(NewLine())
            if self.project_info['other_info']:
                fl.append(self.project_info['other_info'])
            fl.append(NoEscape(r'\vfill'))

            # Stamp
            if self.project_info['stamp_style'] == 'Placeholder':
                fl.append(NoEscape(r'\begin{tikzpicture}'))
                fl.append(NoEscape(r'\draw (0,0) circle (1in);'))
                fl.append(NoEscape(r'\draw (0,0) circle (0.97in);'))
                fl.append(NoEscape(r'\draw (0,0) circle (0.75in);'))
                fl.append(NoEscape(r'\end{tikzpicture}'))
            elif self.project_info['stamp_style'] == 'Yes':
                fl.append(NoEscape(r'\includegraphics[height=2in]{' + self.project_info['stamp_file'] + '}'))
            fl.append(NoEscape(r'\vfill'))
            fl.append(NoEscape(r'\hrule'))
            fl.append(NoEscape(r'\normalsize'))
            fl.append(NewPage())

    def generate_pdf(self):
        file_name = os.path.join(self.project_info['output_folder'], self.project_info['report_name'])
        self.doc.generate_pdf(file_name, clean_tex=False)


class ReportSections:
    def __init__(self):
        pass

    @staticmethod
    def set_parent_section(sections_dict, parent_section, sub_sections_list):
        for sub in sub_sections_list:
            sections_dict[sub]['parent'] = parent_section


class EquipmentReport(Report):

    def __init__(self, project_info, models, pool=None):
        super().__init__(project_info, pool=pool)
        self.models=models
        # self.governing_items = governing_items
        # self.group_dict = group_dict
        self.item_sections_dict = self.get_item_sections_dict()
        if project_info['use_parallel_processing']:
            print('Creating figures with multi-threading')
            self.plots_dict = self.generate_plots(self.pool)
        else:
            print('Creating figures')
            self.plots_dict = self.generate_plots_serial()

        print('Compiling report')
        self.generate_report()
        self.generate_pdf()

    def get_item_sections_dict(self):
        item_sections_dict = {
            mrec.report_section_name: EquipmentReportSections(
                mrec,
                group_name=mrec.group,
                group_idx = mrec.index_in_group)
            for eid, mrec in self.models.items() if mrec.for_report}

        # for equipment_id, model_record in self.models.items():
        #
        #     item = self.results[equipment_id]
        #     if False: #group:
        #         sec_name = group
        #         group_items = [self.results[eq_id] for eq_id in self.group_dict[group]]
        #     else:
        #         sec_name = equipment_id
        #         group_items = None
        #         sec_name = sec_name.rstrip()
        #         if item.equipment_type:
        #             sec_name += f' [{item.equipment_type}]'
        #
        #     item_sections_dict[sec_name] = EquipmentReportSections(item,
        #                                                            group_name=group,
        #                                                            group_items=group_items,
        #                                                            group_idx=governing_idx)
        return item_sections_dict

    def generate_plots_serial(self):
        plots_dict = {}
        for sec_title, report_section in self.item_sections_dict.items():
            for sub_title, sec_pars in {k: v for k, v in report_section.sections_dict.items() if v['include']}.items():
                func_name = sec_pars['section_function'].__name__
                plots_to_generate = EquipmentReportSections.SECTION_PLOTS.get(func_name, {})
                for fig_name, plot_func in plots_to_generate.items():
                    filename = make_figure_filename(sec_title, sub_title, fig_name)
                    if plot_func in plots.VTK_PLOTS:
                        # fig, width = plot_func(report_section.model_record.model, *sec_pars['args'], filename=filename)
                        fig, width = plot_func(report_section.model_record, *sec_pars['args'])
                        file = plots.vtk_save(fig, filename=filename)
                    else:
                        # fig, width = plot_func(report_section.model_record.model, *sec_pars['args'])
                        fig, width = plot_func(report_section.model_record, *sec_pars['args'])
                        file = plots.plt_save(filename=filename)
                    plots_dict[filename] = (width, file)
        return plots_dict

    @staticmethod
    def plots_worker_task(args):
        """Worker function for multiprocessing with profiling"""
        sec_title, sub_title, fig_name, report_section, sec_pars = args
        func_name = sec_pars['section_function'].__name__

        # start_time = time.time()
        filename = make_figure_filename(sec_title, sub_title, fig_name)

        if plot_func := EquipmentReportSections.SECTION_PLOTS.get(func_name, {}).get(fig_name):
            if plot_func in plots.VTK_PLOTS:
                fig, width = plot_func(report_section.item, *sec_pars['args'], filename=filename)
                # plot_time = time.time() - start_time
                # print(f"{filename} PLOTTED in {plot_time:.4f} seconds")
                file = plots.vtk_save(fig, filename=filename)
                # save_time = time.time() - start_time - plot_time
                # print(f"{filename} SAVED in {save_time:.4f} seconds")
            else:
                fig, width = plot_func(report_section.item, *sec_pars['args'])
                # plot_time = time.time() - start_time
                # print(f"{filename} PLOTTED in {plot_time:.4f} seconds")
                file = plots.plt_save(filename=filename)
                # save_time = time.time() - start_time - plot_time
                # print(f"{filename} SAVED in {save_time:.4f} seconds")

        # elapsed_time = time.time() - start_time
        # print(f"Worker {os.getpid()} processed {filename} in {elapsed_time:.4f} seconds")

        return filename, (width, file)

    def generate_plots(self, pool):
        """ Creates figures using multiprocessing and returns a dictionary of figure names and file paths
        """
        plot_args = [
            (sec_name, sub_sec, fig_name, report_section, sec_pars)
            for sec_name, report_section in self.item_sections_dict.items()
            for sub_sec, sec_pars in report_section.sections_dict.items() if sec_pars['include']
            for fig_name in EquipmentReportSections.SECTION_PLOTS.get(sec_pars['section_function'].__name__, {}).keys()
        ]

        results = pool.map(self.plots_worker_task, plot_args)

        plots_dict = {filename: data for filename, data in results if data is not None}

        return plots_dict

    def generate_report(self):
        self.cover_page()
        self.doc.append(NoEscape(r'\pagestyle{StyleSectionSheet}'))
        self.doc.append(NoEscape(r'\tableofcontents'))
        self.doc.append(NewPage())

        section_type = (Section, Subsection, Subsubsection)

        narrative_file = os.path.join(self.project_info['auxiliary_folder'],
                                             self.project_info['narrative_section']) \
            if self.project_info['narrative_section'] else None

        if narrative_file is not None:
            with self.doc.create(Section("Narrative")) as sec:
                insert_framed_pdf(sec, narrative_file)

        active_sections = [None, None, None, None]
        for sec_title, report_section in self.item_sections_dict.items():
            mrec = report_section.model_record
            self.doc.append(NewPage())

            with self.doc.create(Section(sec_title)) as main_sec:
                active_sections[0] = main_sec  # Ensure we store the actual section object

                for sub_title, sec_pars in {k: v for k, v in report_section.sections_dict.items() if
                                            v['include']}.items():
                    title_text = sub_title if sec_pars['alt_title'] is None else sec_pars['alt_title']
                    depth = sec_pars['depth']
                    func = sec_pars['section_function']
                    args = sec_pars['args']
                    kwargs = sec_pars['kwargs']
                    parent = active_sections[depth - 1]

                    with parent.create(section_type[depth](NoEscape(title_text))) as sec:
                        active_sections[depth] = sec
                        func(mrec, sec, sec_title, sub_title, self.plots_dict, *args, **kwargs)


class EquipmentReportSections(ReportSections):
    SECTION_PLOTS = {}

    @classmethod
    def initialize_plots_list(cls):
        cls.SECTION_PLOTS = {
            section_header.__name__: {},
            frontmatter.__name__: {},
            endmatter.__name__: {},
            group_summary.__name__: {},
            description.__name__: {'plan': plots.equipment_plan_view,
                                   '3d': plots.equipment_3d_view_vtk,
                                   'elev_xz': plots.equipment_elevation_view},
            equipment_loads.__name__: {},
            fp_asce7_16.__name__: {},
            fp_cbc_1998.__name__: {},
            fp_asce7_22_opm.__name__: {},
            lrfd_loads_asce7_16.__name__: {},
            lrfd_loads_cbc_1998.__name__: {},
            asd_loads.__name__: {},
            base_anchor_demands.__name__: {'anchor_forces': plots.base_anchors_vs_theta,
                                           'displaced_shape': plots.base_displaced_shape,
                                           'plan': plots.base_equilibrium},
            base_connection_demands.__name__: {},
            wall_brackets.__name__: {},
            wall_bracket_demands.__name__: {'bracket_forces': plots.bracket_vs_theta,
                                            'displaced_shape': plots.bracket_displaced_shape},
            wall_bracket_checks.__name__: {},
            wall_anchor_demands.__name__: {'anchor_forces': plots.wall_anchors_vs_theta,
                                           'displaced_shape': plots.wall_displaced_shape,
                                           'backing': plots.wall_backing},
            bracket_connection_demands.__name__: {},
            sms_connection_demands.__name__: {'sms': plots.sms_hardware_attachment},
            sms_checks.__name__: {},
            wood_fastener_checks.__name__: {},
            concrete_summary_spacing_only.__name__: {'spacing_crit': plots.anchor_spacing_criteria},
            concrete_summary_full.__name__: {'diagram': plots.anchor_basic,
                                             'spacing_crit': plots.anchor_spacing_criteria,
                                             'interaction': plots.anchor_tension_shear_interaction},
            anchor_tension.__name__: {},
            tension_breakout.__name__: {'diagram': plots.anchor_N_breakout},
            tension_pullout.__name__: {},
            side_face_blowout.__name__: {},
            bond_strength.__name__: {},
            anchor_shear.__name__: {},
            shear_breakout.__name__: {'diagram': plots.anchor_V_breakout},
            shear_pryout.__name__: {}
        }

    def __init__(
            self,
            model_record: ModelRecord,
            group_name=None, group_items=None, group_idx=0):
        super().__init__()
        self.model_record = model_record
        self.group_name = group_name
        self.group_items = group_items
        self.group_idx = group_idx
        self.governing_backing = None
        self.wall_anchors = None

        self.sections_dict = self.initialize_sections_dictionary()
        self.set_section_inclusions()
        self.set_section_titles()
        self.set_section_args()

    @staticmethod
    def initialize_sections_dictionary():
        # Define Section List ('Title', function, depth)
        sections_list = [
            # GENERAL SECTIONS
            ('Introduction', frontmatter, 1),
            ('Group Summary', group_summary, 1),
            ('Unit Summary', description, 1),
            ('WARNING: Model Instability', model_instability, 1),
            ('Equipment Loads', equipment_loads, 1),
            ('Seismic Load $F_p$ (ASCE 7-16)', fp_asce7_16, 2),
            ('Maximum Assumed Seismic Load $F_p$', fp_asce7_22_opm, 2),
            ('Seismic Load $F_p$ (CBC 1998)', fp_cbc_1998, 2),
            ('LRFD Factored Loads', lrfd_loads_asce7_16, 2),
            ('LRFD Factored Loads (CBC 1998)', lrfd_loads_cbc_1998, 2),
            ('ASD Factored Loads (CBC 1998)', asd_loads_cbc_1998, 2),
            ('ASD Factored Loads', asd_loads, 2),
            # BASE ANCHOR DEMAND
            ('Base Anchor Demands', base_anchor_demands, 1),
            # BASE CONCRETE ANCHORS
            ('Base Concrete Anchor Checks', section_header, 1),
            ('Base Anchor Spacing Limits', concrete_summary_spacing_only, 2),
            ('Base Concrete Anchor Summary', concrete_summary_full, 2),
            ('Base Anchor in Tension [ACI318-19, 17.6.1]', anchor_tension, 2),
            ('Base Anchor Tension Breakout [ACI318-19, 17.6.2]', tension_breakout, 2),
            ('Base Anchor Pullout [ACI318-19, 17.6.3]', tension_pullout, 2),
            ('Base Anchor Side-Face Blowout [ACI318-19, 17.6.4]', side_face_blowout, 2),
            ('Base Anchor Bond Strength [ACI318-19, 17.6.5]', bond_strength, 2),
            ('Base Anchor in Shear [ACI318-19, 17.7.1]', anchor_shear, 2),
            ('Base Anchor Shear Breakout [ACI318-19, 17.7.2]', shear_breakout, 2),
            ('Base Anchor Shear Pryout [ACI318-19, 17.7.3]', shear_pryout, 2),
            # BASE WOOD ANCHORS
            ('Base Wood Anchor Checks', None, 1),
            # BASE PLATE CONNECTION DEMAND
            ('Base Plate Connections', base_connection_demands, 1),
            ('Base Connection Weld Demand', None, 2),
            ('Base Connection Weld Checks', None, 2),
            ('Base Connection Bolt Demand', None, 2),
            ('Base Connection Bolt Checks', None, 2),
            ('Base Connection SMS Demand', sms_connection_demands, 2),
            ('Base Connection SMS Checks', sms_checks, 2),
            # BASE STRAPS
            ('Base Straps', base_straps, 1),
            # WALL BRACKETS
            ('Wall Brackets', wall_brackets, 1),
            ('Wall Bracket Demand', wall_bracket_demands, 2),
            ('Wall Bracket Checks', wall_bracket_checks, 2),
            # WALL ANCHOR DEMANDS
            ('Wall Fastener Demand', wall_anchor_demands, 1),
            # WALL CONCRETE ANCHORS
            ('Wall Concrete Anchor Checks', section_header, 1),
            ('Wall Anchor Spacing Limits', concrete_summary_spacing_only, 2),
            ('Wall Concrete Anchor Summary', concrete_summary_full, 2),
            ('Wall Anchor in Tension [ACI318-19, 17.6.1]', anchor_tension, 2),
            ('Wall Anchor Tension Breakout [ACI318-19, 17.6.2]', tension_breakout, 2),
            ('Wall Anchor Pullout [ACI318-19, 17.6.3]', tension_pullout, 2),
            ('Wall Anchor Side-Face Blowout [ACI318-19, 17.6.4]', side_face_blowout, 2),
            ('Wall Anchor Bond Strength [ACI318-19, 17.6.5]', bond_strength, 2),
            ('Wall Anchor in Shear [ACI318-19, 17.7.1]', anchor_shear, 2),
            ('Wall Anchor Shear Breakout [ACI318-19, 17.7.2]', shear_breakout, 2),
            ('Wall Anchor Shear Pryout [ACI318-19, 17.7.3]', shear_pryout, 2),
            # WALL CMU ANCHORS
            ('Wall CMU Anchor Checks', cmu_summary_full, 1),
            # WALL SMS ANCHORS
            ('Wall SMS Checks', sms_checks, 2),
            # WALL WOOD FASTENERS
            ('Wall Fastener Checks', wood_fastener_checks, 2),
            # WALL BRACKET CONNECTION
            ('Bracket Connections', bracket_connection_demands, 1),
            ('Bracket Connection SMS Demand', sms_connection_demands, 2),
            ('Bracket Connection SMS Checks', sms_checks, 2),
            ('Addendum', endmatter, 1)]

        # Initialize Dictionary
        sections_dict = {section_name: {'include': False,
                                        'parent': 'Main Section',
                                        'depth': depth,
                                        'sec_obj': None,
                                        'plots': (),
                                        'section_function': func_name,
                                        'alt_title': None,
                                        'args': (),
                                        'kwargs': {}} for section_name, func_name, depth in sections_list}
        return sections_dict

    def set_section_inclusions(self):
        model = self.model_record.model
        sd = self.sections_dict
        code = model.code_pars.code_edition
        run = self.model_record.analysis_runs[self.model_record.governing_run]

        # Front and End Matter
        # sd['Introduction']['include'] = bool(item.frontmatter_file)
        # sd['Addendum']['include'] = bool(item.endmatter_file)

        # Group Summary
        # sd['Group Summary']['include'] = self.group_items is not None

        # Basic Report Elements
        sd['Unit Summary']['include'] = True

        # if self.item.model_unstable:
        #     sd['WARNING: Model Instability']['include'] = True
        #     return
        #
        if not run.omit_analysis:
            # Load Sections
            sd['Equipment Loads']['include'] = True

            # sd['Seismic Load $F_p$ (ASCE 7-16)']['include'] = code == 'ASCE 7-16'
            # sd['Maximum Assumed Seismic Load $F_p$']['include'] = code == 'ASCE 7-22 OPM'
            sd['Seismic Load $F_p$ (CBC 1998)']['include'] = code == 'CBC 1998, 16B'

            if FactorMethod.lrfd in model.analysis_vars.factor_methods:
                # sd['LRFD Factored Loads']['include'] = code in ['ASCE 7-16', 'ASCE 7-22 OPM']
                sd['LRFD Factored Loads (CBC 1998)']['include'] = code == 'CBC 1998, 16B'

            if FactorMethod.asd in model.analysis_vars.factor_methods:
                sd['ASD Factored Loads (CBC 1998)']['include'] = code == 'CBC 1998, 16B'
                # sd['ASD Factored Loads']['include'] = False


            # Base Anchor Demands
            sd['Base Anchor Demands']['include'] = bool(model.elements.base_anchors)

        # Base Concrete Anchor Checks
        has_base_concrete_anchors = bool(model.elements.base_anchors) \
                                    and (model.install.base_material==m.BaseMaterial.concrete)

        if has_base_concrete_anchors:
            sd['Base Concrete Anchor Checks']['include'] = True
            if not all([ba.spacing_requirements.ok for ba in run.results.base_anchors]):
                sd['Base Anchor Spacing Limits']['include'] = True
            elif not model.analysis_vars.omit_analysis:
                ba, ba_idx = get_governing_result(run.results.base_anchors)
                sd['Base Concrete Anchor Summary']['include'] = True
                tg_idx = ba.governing_tension_group
                sg_idx = ba.governing_shear_group

                sd['Base Anchor in Tension [ACI318-19, 17.6.1]']['include'] = (ba.steel_tension_calcs[tg_idx] is not None)
                sd['Base Anchor Tension Breakout [ACI318-19, 17.6.2]']['include'] = (ba.tension_breakout_calcs[tg_idx] is not None)
                sd['Base Anchor Pullout [ACI318-19, 17.6.3]']['include'] = (ba.anchor_pullout_calcs[tg_idx] is not None)
                sd['Base Anchor Side-Face Blowout [ACI318-19, 17.6.4]']['include'] = (ba.side_face_blowout_calcs[tg_idx] is not None)
                sd['Base Anchor Bond Strength [ACI318-19, 17.6.5]']['include'] = (ba.bond_strength_calcs[tg_idx] is not None)
                sd['Base Anchor in Shear [ACI318-19, 17.7.1]']['include'] = (ba.steel_shear_calcs[tg_idx] is not None)
                if ba.shear_breakout_calcs:
                    sd['Base Anchor Shear Breakout [ACI318-19, 17.7.2]']['include'] = (ba.shear_breakout_calcs[sg_idx] is not None)
                sd['Base Anchor Shear Pryout [ACI318-19, 17.7.3]']['include'] = (ba.shear_pryout_calcs[tg_idx] is not None)


        # # Base Connection Demands
        # has_base_connections = len([plate.connection_forces for plate in item.floor_plates if
        #                             not all([plate.x0 == 0, plate.y0 == 0, plate.z0 == 0])]) > 0

        has_base_connections = bool(model.elements.base_plate_connections)

        if has_base_connections and not run.omit_analysis:
            sd['Base Plate Connections']['include'] = True

            # Base SMS Attachment
            sd['Base Connection SMS Demand']['include'] = sd['Base Connection SMS Checks']['include'] = \
                any([isinstance(anchor_obj, sms.SMSAnchors) for anchor_obj in model.elements.base_plate_fasteners])

        # # Base Straps
        # sd['Base Straps']['include'] = len(item.base_straps) > 0 and not self.item.omit_analysis
        #
        # # Wall Brackets and Wall Anchors
        # has_brackets = item.installation_type in ['Wall Brackets', 'Wall Mounted']
        # if has_brackets and not self.item.omit_analysis:
        #     sd['Wall Fastener Demand']['include'] = True
        #     sd['Wall Brackets']['include'] = \
        #         sd['Wall Bracket Demand']['include'] = \
        #         sd['Wall Bracket Checks']['include'] = not self.item.omit_bracket_output
        #     # sd['Wall Anchors']['include'] = \
        #     self.governing_backing = max(item.wall_backing, key=lambda obj: obj.anchor_forces[:, 0].max())
        #
        # Wall Concrete Anchors
        has_wall_concrete_anchors = bool(model.elements.wall_anchors) \
                                    and (model.install.wall_material == m.WallMaterial.concrete)
        if has_wall_concrete_anchors:
            sd['Wall Concrete Anchor Checks']['include'] = True
            if not all([ba.spacing_requirements.ok for ba in run.results.wall_anchors]):
                sd['Wall Anchor Spacing Limits']['include'] = True
            elif not model.analysis_vars.omit_analysis:
                ba, ba_idx = get_governing_result(run.results.wall_anchors)
                sd['Wall Concrete Anchor Summary']['include'] = True
                tg_idx = ba.governing_tension_group
                sg_idx = ba.governing_shear_group

                sd['Wall Anchor in Tension [ACI318-19, 17.6.1]']['include'] = (ba.steel_tension_calcs[tg_idx] is not None)
                sd['Wall Anchor Tension Breakout [ACI318-19, 17.6.2]']['include'] = (ba.tension_breakout_calcs[tg_idx] is not None)
                sd['Wall Anchor Pullout [ACI318-19, 17.6.3]']['include'] = (ba.anchor_pullout_calcs[tg_idx] is not None)
                sd['Wall Anchor Side-Face Blowout [ACI318-19, 17.6.4]']['include'] = (ba.side_face_blowout_calcs[tg_idx] is not None)
                sd['Wall Anchor Bond Strength [ACI318-19, 17.6.5]']['include'] = (ba.bond_strength_calcs[tg_idx] is not None)
                sd['Wall Anchor in Shear [ACI318-19, 17.7.1]']['include'] = (ba.steel_shear_calcs[tg_idx] is not None)
                if ba.shear_breakout_calcs:
                    sd['Wall Anchor Shear Breakout [ACI318-19, 17.7.2]']['include'] = (ba.shear_breakout_calcs[sg_idx] is not None)
                sd['Wall Anchor Shear Pryout [ACI318-19, 17.7.3]']['include'] = (ba.shear_pryout_calcs[tg_idx] is not None)
        #
        # # Wall CMU Anchors
        # has_wall_cmu_anchors = any(
        #     [isinstance(anchors, CMUAnchors) for wall, anchors in item.wall_anchors.items()])
        # if has_wall_cmu_anchors and not self.item.omit_analysis:
        #     sd['Wall CMU Anchor Checks']['include'] = True
        #
        # # Wall SMS Anchors
        # has_wall_sms_anchors = any([isinstance(b.anchors_obj, SMSAnchors)
        #                             for b in item.wall_backing])
        # if has_wall_sms_anchors and not self.item.omit_analysis:
        #     self.wall_anchors = self.governing_backing.anchors_obj
        #     sd['Wall SMS Checks']['include'] = True
        #
        # # Wall Wood Fasteners
        # has_wall_wood_fasteners = any([isinstance(b.anchors_obj, WoodFastener)
        #                             for b in item.wall_backing])
        # if has_wall_wood_fasteners and not self.item.omit_analysis:
        #     self.wall_anchors = self.governing_backing.anchors_obj
        #     sd['Wall Fastener Checks']['include'] = True
        #
        #
        # Bracket Connection Demands
        if model.elements.wall_bracket_connections and not model.analysis_vars.omit_analysis:
            sd['Bracket Connections']['include'] = True

            # Bracket SMS Attachment
            sd['Bracket Connection SMS Demand']['include'] = sd['Bracket Connection SMS Checks']['include'] = \
                any([isinstance(anchor_obj, sms.SMSAnchors) for anchor_obj in model.elements.wall_bracket_fasteners])

    def set_section_titles(self):
        run = self.model_record.analysis_runs[self.model_record.governing_run]
        if self.sections_dict['Wall Brackets']['include']:
            self.sections_dict['Wall Brackets']['alt_title'] = f'{run.hardware_selection.bracket_id} (Wall Bracket)'
            self.sections_dict['Wall Bracket Demand']['alt_title'] = f'{run.hardware_selection.bracket_id} Demand'
            self.sections_dict['Wall Bracket Checks']['alt_title'] = f'{run.hardware_selection.bracket_id} Checks'

    def set_section_args(self):
        """ This provides any additional arguments used by the section functions beyond the standard
        item, sec, sec_title, sub_title, plots_dict"""

        model = self.model_record.model
        sd = self.sections_dict
        run = self.model_record.analysis_runs[self.model_record.governing_run]

        # Group Summary
        sd['Group Summary']['args'] = (self.group_name, self.group_items, self.group_idx)

        # Base Concrete Anchors
        conc_sections_list = ['Anchor Spacing Limits',
                              'Concrete Anchor Summary',
                              'Anchor in Tension [ACI318-19, 17.6.1]',
                              'Anchor Tension Breakout [ACI318-19, 17.6.2]',
                              'Anchor Pullout [ACI318-19, 17.6.3]',
                              'Anchor Side-Face Blowout [ACI318-19, 17.6.4]',
                              'Anchor Bond Strength [ACI318-19, 17.6.5]',
                              'Anchor in Shear [ACI318-19, 17.7.1]',
                              'Anchor Shear Breakout [ACI318-19, 17.7.2]',
                              'Anchor Shear Pryout [ACI318-19, 17.7.3]']

        if run.results.base_anchors:
            ba_res, ba_idx = get_governing_result(run.results.base_anchors)
            ba_obj = model.elements.base_anchors[ba_idx]
            for limit_name in conc_sections_list:
                sd['Base ' + limit_name]['args'] = (ba_obj, ba_res)

        # Base SMS Connection
        if sd['Base Connection SMS Demand']['include']:
            sms_result, governing_sms_idx = get_governing_result(run.results.base_plate_fasteners)
            sms_obj = model.elements.base_plate_fasteners[governing_sms_idx]
            governing_cxn_idx = model.elements.sms_to_cxn[governing_sms_idx]
            cxn_obj = model.elements.base_plate_connections[governing_cxn_idx]
            cxn_res = run.results.base_plate_connections[governing_cxn_idx]
            # base_connection_obj = max([plate.connection for plate in item.floor_plates if plate.connection is not None],
            #                           key=lambda x: x.anchors_obj.max_dcr())
            sd['Base Plate Connections']['args'] = (governing_cxn_idx,)
            sd['Base Connection SMS Demand']['args'] = (cxn_obj, cxn_res, sms_obj, sms_result, 'base')
            sd['Base Connection SMS Checks']['args'] = (sms_obj, sms_result)

        # Wall Anchor Demands
        sd['Wall Fastener Demand']['args'] = (self.governing_backing,)

        # Wall Concrete Anchors
        wa_res, wa_idx = get_governing_result(run.results.wall_anchors)
        wa_obj = model.elements.wall_anchors[wa_idx]
        for limit_name in conc_sections_list:
            sd['Wall ' + limit_name]['args'] = (wa_obj, wa_res)

        # Wall SMS Anchors
        sd['Wall SMS Checks']['args'] = (self.wall_anchors,)

        # Wall Wood Fasteners
        sd['Wall Fastener Checks']['args'] = (self.wall_anchors,)

        # Bracket SMS Connection
        if sd['Bracket Connections']['include']:
            sms_result, governing_sms_idx = get_governing_result(run.results.wall_bracket_fasteners)
            sms_obj = model.elements.wall_bracket_fasteners[governing_sms_idx]
            governing_cxn_idx = governing_sms_idx
            cxn_obj = model.elements.wall_bracket_connections[governing_cxn_idx]
            cxn_res = run.results.wall_bracket_connections[governing_cxn_idx]
            # base_connection_obj = max([plate.connection for plate in item.floor_plates if plate.connection is not None],
            #                           key=lambda x: x.anchors_obj.max_dcr())

            sd['Bracket Connection SMS Demand']['args'] = (cxn_obj, cxn_res, sms_obj, sms_result, 'bracket')
            sd['Bracket Connection SMS Checks']['args'] = (sms_obj, sms_result)

        # Unity Description and Summary
        inclusions_summary = {sec: self.sections_dict[sec]['include'] for sec in self.sections_dict}
        sd['Unit Summary']['args'] = (inclusions_summary,)

def section_header(*args, **kwargs):
    pass


def frontmatter(item, sec, sec_title, sub_title, plots_dict):
    insert_framed_pdf(sec, item.frontmatter_file, subsection_title='Narrative')


def endmatter(item, sec, sec_title, sub_title, plots_dict):
    insert_framed_pdf(sec, item.endmatter_file, subsection_title='Addendum')


def group_summary(item, sec, sec_title, sub_title, plots_dict, group_name, group_items, governing_index):
    sec.append("This section is applicable to the equipment items tabulated below. Calculations are presented for "
               "the configuration which results in maximum anchor tension. Similar calculations were completed "
               r"for all units, but are omitted from this report for brevity.")
    sec.append(NoEscape(r'\bigskip'))
    sec.append(NewLine())

    # Group Summary Table
    title = f'Group {group_name} Summary'
    header = ['Equipment ID', 'Equipment Type',
              NoEscape('$B_x$'), NoEscape('$B_y$'), NoEscape('$H$'), NoEscape('$z_{CG}$'),
              NoEscape('$W_p$'), NoEscape('$E_h$'), NoEscape('$T_{max}$'), 'DCR']
    units = ['', '', '(in)', '(in)', '(in)', '(in)', '(lbs)', '(lbs)', '(lbs)', '']
    utilization_column = [len(header)-1]
    alignment = 'p{1.25in}p{1.25in}cccccccc'
    if item.installation_type in ['Base Anchored']:
        data = [[item.equipment_id, item.equipment_type,
                 item.Bx, item.By, item.H, item.zCG, item.Wp, item.Eh, item.base_anchors.Tu_max, item.base_anchors.DCR] for item in
                group_items]

    elif item.installation_type in ['Wall Brackets', 'Wall Mounted']:

        anchor_tensions = [max([anchors.Tu_max for wall, anchors in item.wall_anchors.items() if anchors is not None]+
                               [b.anchors_obj.Tu_max for b in item.wall_backing if b.anchors_obj is not None]) for item
                           in group_items]
        dcrs = [max([anchors.DCR for wall, anchors in item.wall_anchors.items() if anchors is not None]+
                               [b.anchors_obj.DCR for b in item.wall_backing if b.anchors_obj is not None]) for item
                           in group_items]
        data = [[item.equipment_id, item.equipment_type,
                 item.Bx, item.By, item.H, item.zCG, item.Wp, item.Eh, tension, dcr] for item, tension, dcr in
                zip(group_items, anchor_tensions, dcrs)]
    else:
        raise Exception("Installation Type Not Supported")

    formats = ['{}', '{}', '{:.0f}', '{:.0f}', '{:.0f}', '{:.0f}', '{:.0f}', '{:.0f}', '{:.0f}', '{:.2f}']
    make_table(sec, title, header, units, data, col_formats=formats,
               rows_to_highlight=governing_index, utilization_cols=utilization_column, alignment=alignment,
               use_minipage=False)


def description(
        model_record: ModelRecord,
        sec: Union[Section, Subsection, Subsubsection],
        sec_title: str,
        sub_title: str,
        plots_dict: dict,
        inclusions_dict: dict):

    #Unpack Data
    model = model_record.model
    eprops = model.equipment_props
    run = model_record.analysis_runs[model_record.governing_run]

    #Create Section
    with sec.create(Tabularx('lX', pos='t')) as table:
        table.add_hline()
        header = [NoEscape(r'\rowcolor{lightgray} Unit Parameters'), '']
        table.add_row(header)
        table.add_hline()
        table.add_row(['Equipment ID', model.equipment_info.equipment_id])
        table.add_hline()
        table.add_row([NoEscape('Max Operating Weight, $W_p$'), rf'{eprops.Wp:.2f} lbs'])
        table.add_row([NoEscape(r'Dimensions, $W\times D\times H$'),
                       NoEscape(rf'${eprops.Bx:.2f}$ in. $\times {eprops.By:.2f}$ in. $\times{eprops.H:.2f}$ in.')])
        if eprops.ex == eprops.ey == 0.0:
            table.add_row([NoEscape(r'Center of Gravity, $z_{CG}$'), NoEscape(rf'${eprops.zCG:.2f}$ in.')])
        else:
            table.add_row([NoEscape(r'Center of Gravity, $e_x$, $e_y$, $z_{CG}$'),
                           NoEscape(rf'${eprops.ex:.2f}$ in.,  ${eprops.ey:.2f}$ in., ${eprops.zCG:.2f}$ in.')])
        table.add_hline()
        if model.elements.base_anchors and isinstance(model.elements.base_anchors[0], conc.ConcreteAnchors):
            ba_res, ba_idx = get_governing_result(run.results.base_anchors)
            ba = model.elements.base_anchors[ba_idx]
            ba_calc = ba_res.tension_breakout_calcs[ba_res.governing_tension_group]
            table.add_row([NoEscape(r'\rowcolor{lightgray} Base Anchor and Substrate'), ''])
            table.add_hline()
            table.add_row(['Base Condition', 'Anchorage to Concrete'])
            table.add_row(['Anchor Type', NoEscape(rf'\textbf{{{run.hardware_selection.base_anchor_id}}}')])
            table.add_row([NoEscape('$h_{ef}$'), rf'{ba_calc.hef_default}'])
            cracked_text = 'Cracked Concrete, ' if ba.concrete_props.cracked_concrete else 'Uncracked Concrete, '
            fc_text = rf'$f^\prime_c = {ba.concrete_props.fc:.0f}$ psi'
            table.add_row(['Base Material', NoEscape(cracked_text + fc_text)])
            table.add_row(['Base Thickness', f'{ba.concrete_props.t_slab:.2f}'])
            # if model.include_pull_test and not model.omit_analysis:
            #     model.update_element_resultants(model.governing_solutions['base_anchor_tension']['sol'])
            #     table.add_row(['Pull-test Load', f'{max([500, 3 * model.base_anchors.Tu_max]):.0f} lbs'])
            # if model.base_straps:
            #     table.add_row(['Base Strap', model.base_strap])
            table.add_hline()
        if model.elements.wall_brackets:
            wa_res, wa_idx = get_governing_result(run.results.wall_anchors)
            wa = model.elements.wall_anchors[wa_idx]
            wa_calc = wa_res.tension_breakout_calcs[wa_res.governing_tension_group]
            table.add_row([NoEscape(r'\rowcolor{lightgray} Wall Fastener and Substrate'), ''])
            table.add_hline()
            table.add_row(['Wall Type', rf'{model.install.wall_material}'])
            if not model_record.omit_bracket_output:
                table.add_row(['Unit-to-Wall Hardware', run.hardware_selection.bracket_id])
            anchor_idx, anchors = get_governing_result(run.results.wall_anchors)
            if model.install.wall_material==WallMaterial.concrete:
                table.add_row(['Wall Fastener', NoEscape(rf'\textbf{{{run.hardware_selection.wall_anchor_id}}}')])
                table.add_row([NoEscape('$h_{ef}$'), rf'{wa_calc.hef_default}'])
                cracked_text = 'Cracked Concrete, ' if wa.concrete_props.cracked_concrete else 'Uncracked Concrete, '
                fc_text = rf'$f^\prime_c = {wa.concrete_props.fc:.0f}$ psi'
                table.add_row(['Wall Material', NoEscape(cracked_text + fc_text)])
                table.add_row(['Wall Thickness', f'{wa.concrete_props.t_slab:.2f}'])
            #
            # elif isinstance(anchor_obj, SMSAnchors):
            #     table.add_row(['Wall Fastener', NoEscape(rf'\textbf{{{model.wall_sms_id}}}')])
            #     table.add_row(['Wall Type', f'Steel Studs, {anchor_obj.gauge:.0f} GA, {anchor_obj.fy:.0f} ksi'])
            # if model.include_pull_test and not model.omit_analysis:
            #     wall_anchors = []
            #     model.update_element_resultants(model.governing_solutions['wall_anchor_tension']['sol'])
            #     pull_test_text = f'{max([3 * anchor_obj.Tu_max, 500]):.0f} lbs' if anchor_obj.Tu_max else 'N/A'
            #     table.add_row(['Pull-test Load', pull_test_text])

            table.add_hline()
    # todo: add ASCE7 classification, once implemented in spreadsheet

    sec.append(NoEscape(r'\bigskip'))
    sec.append(NewLine())
    with sec.create(MiniPage(width=r"6.5in", pos='t')) as mini:
        # Equipment Figure
        # subheader(mini,'Isometric')
        fig_name = '3d'
        width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
        make_figure(mini, 2, file,title='ISOMETRIC')

        mini.append(NoEscape(r'\hfill'))

        if model.elements.base_anchors:
            # Equipment Plan View
            fig_name = 'plan'
            width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
            make_figure(mini, 2, file,title='PLAN')
            mini.append(NoEscape(r'\hfill'))

        fig_name = 'elev_xz'
        width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
        make_figure(mini, 2, file, title='ELEVATION')

    _results_summary(model_record, sec, inclusions_dict)

def _results_summary(model_record, sec,inclusions_dict):
    model = model_record.model
    run = model_record.analysis_runs[model_record.governing_run]
    results = run.results

    def make_element_summary_row(
            element_label, obj, obj_res: RESULTS_LIKE):
        method_str = obj.factor_method[:]
        theta_at_max = np.degrees(run.solutions[obj.factor_method].theta_z[obj_res.governing_theta_idx])
        ok = 'OK' if obj_res.ok else 'NG' #r'\textcolor{Green}{\textbf{\textsf{OK}}}' if obj_res.ok else r'\textcolor{red}{\textbf{\textsf{NG}}}'
        return [element_label, method_str, theta_at_max, obj_res.unity, NoEscape(ok)]

    sec.append(NoEscape(r'\bigskip'))
    sec.append(NewLine())
    data = []

    if inclusions_dict['Base Concrete Anchor Summary']:
        ba_res, ba_idx = get_governing_result(results.base_anchors)
        ba = model.elements.base_anchors[ba_idx]
        data.append(make_element_summary_row("Base Concrete Anchors",ba, ba_res))
    if inclusions_dict['Base Connection SMS Checks']:
        sms_res, sms_idx = get_governing_result(results.base_plate_fasteners)
        sms_obj = model.elements.base_plate_fasteners[sms_idx]
        data.append(make_element_summary_row("Base Plate Connection", sms_obj, sms_res))

    title = 'Summary of Element Checks'
    header = [NoEscape('Element'), NoEscape('Design Method'), NoEscape('Load Direction at Max Unity'), NoEscape('Max Unity'), NoEscape('OK?')]
    units = ['', '', '(Deg)', '', '']
    formats = ['{}', '{}', '{:.0f}', '{:.2f}', '{}']
    make_table(sec, title, header, units, data, col_formats=formats, utilization_cols=[3,4], width=r'\textwidth',
               use_minipage=True, align='l')


def equipment_loads(
        model_record: ModelRecord,
        sec: Union[Section, Subsection, Subsubsection],
        sec_title: str,
        sub_title: str,
        plots_dict: dict):
    """ Placeholder for subsections"""
    pass


def fp_asce7_16(item, sec, sec_title, sub_title, plots_dict):
    ap = item.code_pars['ap']
    Rp = item.code_pars['Rp']
    Ip = item.code_pars['Ip']
    sds = item.code_pars['sds']
    z = item.code_pars['z']
    h = item.code_pars['h']
    omega = item.code_pars['omega']
    use_dynamic = item.code_pars['use_dynamic']
    ai = item.code_pars['ai']
    Ax = item.code_pars['Ax']

    with sec.create(Flalign(numbering=False, escape=False)) as align:
        align.append(rf'''&F_{{p,code}}
            &&=\frac{{0.4a_pS_{{DS}}W_p}}{{\left(\frac{{R_p}}{{I_p}}\right)}}\left(1+2\frac{{z}}{{h}}\right)
            &&=\frac{{0.4({ap:.2f})({sds:.2f})({item.Wp:.2f})}}
            {{\left(\frac{{({Rp:.2f})}}{{({Ip:.2f})}}\right)}}
            \left(1+2\frac{{({z:.2f})}}{{({h:.2f})}}\right)
            &&={item.Fp_code:.2f} \text{{ lb}}
            &\quad \text{{ASCE7-16 13.3-1}}\\''')
        align.append(rf'''&F_{{p,min}} 
            &&=0.3S_{{DS}}I_pW_p
            &&=0.3({sds:.2f})({Ip:.1f})({item.Wp:.2f})
            &&={item.Fp_min:.2f} \text{{ lb}}
            &\quad \text{{ASCE7-16 13.3-3}}\\''')
        align.append(rf'''&F_{{p,max}}
            &&=1.6S_{{DS}}I_pW_p
            &&= 1.6({sds:.2f})({Ip:.1f})({item.Wp:.2f})
            &&= {item.Fp_max:.2f} \text{{ lb}}
            &\quad \text{{ASCE7-16 13.3-2}}\\''')
        align.append(rf'''&F_p 
            &&
            &&
            &&= {item.Fp:.2f} \text{{ lb}}
            &\text{{ASCE7-16 13.3.1.1}}\\''')

    with sec.create(Flalign()) as align:
        align.append(
            rf'''&E_h &&= F_p &&= {item.Fp:.2f} \text{{ lb}} &\text{{\hfill Horizontal Seismic Force}}\\''')
        if item.include_overstrength:
            align.append(
                rf'''&E_{{mh}} &&= \Omega F_p &&={item.Emh:.2f} \text{{ lb}} &\text{{Seismic Force with Overstrength}}\\''')
        align.append(
            rf'''&E_v &&= 0.2S_{{DS}}W_p && = {item.Ev:.2f} \text{{ lb}} &\text{{Vertical Seismic Force}}\\''')


def fp_asce7_22_opm(item, sec, sec_title, sub_title, plots_dict):  
    with sec.create(Flalign()) as align:
        align.append(
            rf'''&C_{{pm}} && &&= {item.code_pars["Cpm"]:.2f} &\text{{\hfill Maximum Considered Horizontal Design Force Coefficient}}\\''')
        align.append(
            rf'''&C_{{v}} && &&= {item.code_pars["Cv"]:.2f} &\text{{\hfill Maximum Considered Vertical Design Force Coefficient}}\\''')
        align.append(
            rf'''&E_h &&= F_p = C_{{pm}}W_p &&= {item.Fp:.2f} \text{{ lb}} &\text{{\hfill Horizontal Seismic Force}}\\''')
        if item.include_overstrength:
            align.append(
                rf'''&E_{{mh}} &&= \Omega F_p &&={item.Emh:.2f} \text{{ lb}} &\text{{Seismic Force with Overstrength}}\\''')
        align.append(
            rf'''&E_v &&= C_{{v}}W_p && = {item.Ev:.2f} \text{{ lb}} &\text{{Vertical Seismic Force}}\\''')

def fp_cbc_1998(
        model_record: ModelRecord,
        sec: Union[Section, Subsection, Subsubsection],
        sec_title: str,
        sub_title: str,
        plots_dict: dict):

    model = model_record.model
    calc = model.fp_calc
    with sec.create(Tabular('p{0.35\\textwidth} p{0.6\\textwidth}', pos='t')) as table:
        table.add_hline()
        header = [NoEscape(r'\rowcolor{lightgray} Unit Parameters'), '']
        table.add_row(header)
        table.add_hline()
        table.add_row(['Table 16B-O Category', model.code_pars.cp_category])
        table.add_hline()
        table.add_row([NoEscape('$C_p$'), model.code_pars.Cp])
        table.add_hline()
        table.add_row([NoEscape('Rigid/Flexible Factor, $R$'), model.code_pars.cp_amplification])
        table.add_hline()
        table.add_row([NoEscape(r'At-or-below-grade Factor, $G$'),
                       f'{model.code_pars.grade_factor:.2f}'])
        table.add_hline()

    with sec.create(Flalign(numbering=False, escape=False)) as align:
        align.append(rf'''&C^\prime_{{p}}
            &&=C_p  R  G && \leq \begin{{cases}}
            2G & \text{{for }} R = 2\\
            3G & \text{{for }} R = 4
            \end{{cases}}  && = {model.code_pars.Cp_eff}
            & \text{{CBC98 (\S1630B.2)}}\\''')
        align.append(rf'''&F_{{p}}
                    &&=ZI_pC^\prime_{{p}}W_p &&
                    &&={calc.Fp:.2f} \text{{ lb}}
                    &\quad \text{{CBC98 (30B-1)}}\\''')
        align.append(
            rf'''&E_h &&= F_p && &&= {calc.Eh:.2f} \text{{ lb}} &\text{{\hfill Horizontal Seismic Force}}\\''')
        align.append(
            rf'''&E_v &&= F_p/3 && && = {calc.Ev:.2f} \text{{ lb}} &\text{{Vertical Seismic Force}}''')

    # with sec.create(Flalign(numbering=False, escape=False)) as align:
    #     align.append(rf'''&F_{{p}}
    #         &&=ZI_pC^\prime_{{p}}W_p
    #         &&= \left({item.code_pars['Z']}\right)\left({item.code_pars['Ip']}\right)\left({item.code_pars['Cp_eff']}\right)\left({item.Wp:.2f}\right)
    #         &&={item.Fp:.2f} \text{{ lb}}
    #         &\quad \text{{CBC98 (30B-1)}}''')
    #
    # with sec.create(Flalign()) as align:
    #     align.append(
    #         rf'''&E_h &&= F_p/0.7 &&= {item.Eh:.2f} \text{{ lb}} &\text{{\hfill LRFD Horizontal Seismic Force}}\\''')
    #     align.append(
    #         rf'''&E_v &&= F_p/(0.7\cdot 3) && = {item.Ev:.2f} \text{{ lb}} &\text{{LRFD Vertical Seismic Force}}\\''')


def lrfd_loads_asce7_16(item, sec, sec_title, sub_title, plots_dict):
    with sec.create(Flalign()) as align:
        if item.include_overstrength:
            align.append(NoEscape(
                rf'''&F_{{uh}} &&=1.0E_{{mh}} &&=1.0({item.Emh:.2f}) &&= {item.Emh:.2f} \text{{ lb}} &\text{{ASCE 7 \S2.3.6-6}}\\'''))
        else:
            align.append(NoEscape(
                rf'''&F_{{uh}} &&=1.0E_h &&=1.0({item.Eh:.2f}) &&= {item.Fuh:.2f} \text{{ lb}} &\text{{ASCE 7 \S2.3.6-6}}\\'''))
        align.append(NoEscape(rf'''&F_{{uv,min}} &&=-0.9W_p+1.0E_v &&=-0.9({item.Wp:.2f})+1.0({item.Ev:.2f})
        &&={item.Fuv_min:.2f} \text{{ lb}} & \text{{ASCE 7 \S2.3.6-6}}\\ '''))
        align.append(NoEscape(rf'''&F_{{uv,max}} &&=-1.2W_p-1.0E_v &&=-1.2({item.Wp:.2f})-1.0({item.Ev:.2f})
                            &&={item.Fuv_max:.2f} \text{{ lb}} & \text{{ASCE 7 \S2.3.6-7}} '''))

    _analysis_description_text(sec)
    


def lrfd_loads_cbc_1998(
        model_record: ModelRecord,
        sec: Union[Section, Subsection, Subsubsection],
        sec_title: str,
        sub_title: str,
        plots_dict: dict):
    model = model_record.model
    eprops = model.equipment_props
    fp_calc = model.fp_calc
    fl_calc = model.factored_loads.lrfd

    with sec.create(Flalign()) as align:

        align.append(
            rf'''&F_{{h}} &&=0.75(1.7)(1.1E_h) &&=0.75(1.7)(1.1)({fp_calc.Eh:.2f}) 
            &&= {fl_calc.Fh:.2f} \text{{ lb}} &\text{{CBC98 (9B-2)}}\\''')
        align.append(rf'''&F_{{v,min}} &&=-0.9W_p+1.3(1.1E_v) &&=-0.9({eprops.Wp:.2f})+1.3(1.1)({fp_calc.Ev:.2f})
        &&={fl_calc.Fv_min:.2f} \text{{ lb}} & \text{{CBC98 (9B-3)}}\\ ''')
        align.append(rf'''&F_{{v,max}} &&=-0.75(1.4W_p+1.7(1.1E_v)) &&=-0.75(1.4{eprops.Wp:.2f})-1.7(1.1)({fp_calc.Ev:.2f})
                            &&={fl_calc.Fv_max:.2f} \text{{ lb}} & \text{{CBC98 (9B-2)}} ''')

    _analysis_description_text(sec)

def asd_loads_cbc_1998(
        model_record: ModelRecord,
        sec: Union[Section, Subsection, Subsubsection],
        sec_title: str,
        sub_title: str,
        plots_dict: dict):
    model = model_record.model
    eprops = model.equipment_props
    fp_calc = model.fp_calc
    fl_calc = model.factored_loads.lrfd

    with sec.create(Flalign()) as align:
        align.append(
            rf'''&F_{{h}} &&=1.0E_h &&=1.0({fp_calc.Eh:.2f})) 
               &&= {fl_calc.Fh:.2f} \text{{ lb}} &\text{{CBC98 (  )}}\\''')
        align.append(rf'''&F_{{v,min}} &&=-0.9W_p+1.3(1.1E_v) &&=-0.9({eprops.Wp:.2f})+1.3(1.1)({fp_calc.Ev:.2f})
           &&={fl_calc.Fv_min:.2f} \text{{ lb}} & \text{{CBC98 (  )}}\\ ''')
        align.append(
            rf'''&F_{{uv,max}} &&=-0.75(1.4W_p+1.7(1.1E_v)) &&=-0.75(1.4{eprops.Wp:.2f})-1.7(1.1)({fp_calc.Ev:.2f})
                               &&={fl_calc.Fv_max:.2f} \text{{ lb}} & \text{{CBC98 (  )}} ''')

def _analysis_description_text(sec):
    sec.append(
        NoEscape(r'''The factored horizontal load, $F_{h}$ is applied at angles, $0 \leq \theta_z \leq 360$. 
                An analytical model is used to determine distribution of applied loads to anchoring elements.'''))

def asd_loads(item, sec, sec_title, sub_title, plots_dict):
    #todo: rename and refactor for 7-16 and 7-22
    with sec.create(Flalign()) as align:
        if item.include_overstrength:
            align.append(
                rf'''&F_{{uh}} &&=0.7E_{{mh}} &&=0.7({item.Emh:.2f}) &&= {item.Emh:.2f} \text{{ lb}} &\text{{ASCE7-16 2.4.5-8}}\\''')
        else:
            align.append(
                rf'''&F_{{ah}} &&=0.7E_h &&=0.7({item.Eh:.2f}) &&= {item.Fah:.2f} \text{{ lb}} &\text{{ASCE7-16 2.4.5-8}}\\''')
        align.append(rf'''&F_{{av,min}} &&=-0.6W_p+0.7E_v &&=-0.6({item.Wp:.2f})+0.7({item.Ev:.2f})
        &&={item.Fav_min} \text{{ lb}} & \text{{ASCE7-16 2.4.5-10}} \\''')
        align.append(rf'''&F_{{av,max}} &&=-1.0W_p-0.7E_v &&=-1.0({item.Wp:.2f})-0.7({item.Ev}:.2f)
                            &&={item.Fav_max:.2f} \text{{ lb}} & \text{{ASCE7-16 2.4.5-8}} ''')

    sec.append(
        NoEscape(r'''The factored horizontal load, $F_{ah}$ is applied at angles, $0 \leq \theta_z \leq 360$. 
                An analytical model is used to determine distribution of applied loads to anchoring elements.'''))


def base_anchor_demands(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict):
    """Note, this section is only included if base anchors are present in the model.
    Section code depends on there being at least one base anchors object and corresponding results object"""

    model = model_record.model
    run = model_record.analysis_runs[model_record.governing_run]

    # Anchor Forces Plot
    title = 'Anchor Forces vs. Direction of Loading'
    fig_name = 'anchor_forces'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file, title=title)

    # Displaced Shape Plot
    fig_name = 'displaced_shape'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file)

    # Base Reactions and Governing Load Angle
    sec.append(NoEscape(r'\bigskip'))
    sec.append(NewLine())
    title = 'Base Reactions at Governing Load Angle'
    fig_name = 'plan'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, 2.5, file, title=title)

    sec.append(NoEscape(r'\hfill'))
    with sec.create(MiniPage(width='3in', pos='t', align='r')) as mini:
        ba_res, ba_idx = get_governing_result(run.results.base_anchors)
        theta_idx = ba_res.governing_theta_idx
        cz_states = [res.compression_zones[theta_idx] for res in run.results.base_plates]

        # Anchors Table
        data_blocks = []
        for ba, ba_res in zip(model.elements.base_anchors, run.results.base_anchors):
            xy = ba.geo_props.xy_anchors
            K_val = ba.anchor_props.K
            K_col = np.full((xy.shape[0],1),K_val, dtype=float)
            forces = ba_res.forces[:,:,theta_idx]
            data_blocks.append(np.hstack((xy, K_col, forces)))
        data = np.vstack(data_blocks)
        title = 'Summary of Base Anchors(s)'
        header = [NoEscape('$x$'), NoEscape('$y$'), NoEscape('$K^{(+)}$'), NoEscape('$V_x$'), NoEscape('$V_y$'), NoEscape('$N$')]
        units = ['(in)', '(in)', '(lb/in)', '(lbs)', '(lbs)', '(lbs)']
        formats = ['{:.2f}', '{:.2f}', '{:.0f}', '{:.0f}', '{:.0f}', '{:.0f}']
        make_table(mini, title, header, units, data, col_formats=formats, width=r'\textwidth',
                   use_minipage=True, align='l') #todo: ...,rows_to_highlight=model.base_anchors.group_idx)

        mini.append(NoEscape(r'\bigskip'))
        mini.append(NewLine())

        # Bearing Areas Table
        if model.elements.base_plates:  # todo [Testing]: Verify if this section works with anchors but no bearing boundaries
            # mini.append(NoEscape(r'\bigskip'))
            title = 'Summary of Bearing Area(s)'
            header = [NoEscape(r'$\bar{x}$'),
                      NoEscape(r'$\bar{y}$'),
                      NoEscape(r'$A$'),
                      # NoEscape(r'$I_{xx}$'),
                      # NoEscape(r'$I_{xx}$'),
                      # NoEscape(r'$I_{xy}$'),
                      NoEscape(r'$\beta$'),
                      NoEscape('$N$')]
            units = ['(in)', '(in)',
                     NoEscape(r'(in\textsuperscript{2})'),
                     # NoEscape(r'(in\textsuperscript{4})'),
                     # NoEscape(r'(in\textsuperscript{4})'),
                     # NoEscape(r'(in\textsuperscript{4})'),
                     NoEscape(r'(lb/in\textsuperscript{3})'),
                     '(lbs)']
            data = []
            for plate, cz in zip(model.elements.base_plates, cz_states):
                for index in range(len(cz.areas)):
                    data.append([
                        cz.centroids[index][0],
                        cz.centroids[index][1],
                        cz.areas[index],
                        # cz_result["Ixx"][index],
                        # cz_result["Iyy"][index],
                        # cz_result["Ixy"][index],
                        cz.beta[index],
                        -cz.fz[index]])

            # formats = ['{:.2f}', '{:.2f}', '{:.2f}', '{:.2f}', '{:.2f}', '{:.2f}', '{:.0f}', '{:.0f}']
            formats = ['{:.2f}', '{:.2f}', '{:.2f}', '{:.0f}', '{:.0f}']
            make_table(mini, title, header, units, data, col_formats=formats, width=r'\textwidth', use_minipage=True,align='l')

def base_connection_demands(
        model_record: ModelRecord,
        sec: Union[Section, Subsection, Subsubsection],
        sec_title: str,
        sub_title: str,
        plots_dict: dict,
        governing_cxn_idx=None):
    model = model_record.model
    run = model_record.analysis_runs[model_record.governing_run]
    if run.results.base_plate_fasteners:
        sms_res, sms_idx = get_governing_result(run.results.base_plate_fasteners)
        theta_idx = sms_res.governing_theta_idx
    else:
        ba_res, ba_idx = get_governing_result(run.results.base_anchors)
        theta_idx = ba_res.governing_theta_idx

    # Attachment Forces Table
    data = [np.concatenate([[plate.props.xc, plate.props.yc, plate.props.zc], res.connection_forces[:,theta_idx]])
            for plate, res in zip(model.elements.base_plates, run.results.base_plates)  if
            not all([plate.props.xc == 0, plate.props.yc == 0, plate.props.zc == 0])]

    sec.append(
        "The table below summarizes the internal reaction forces at the attachment points "
        "for base plate elements. Connection of the base plate to the equipment unit "
        "(through welds, fasteners, etc.) must be designed to accommodate these forces.")

    sec.append(NewLine())
    header = [NoEscape('$x$'), NoEscape('$y$'), NoEscape('$z$'),
              NoEscape('$R_x$'),
              NoEscape('$R_y$'),
              NoEscape('$R_z$'),
              NoEscape('$M_x$'),
              NoEscape('$M_y$'),
              NoEscape('$M_z$')]
    units = ['(in)'] * 3 + ['(lbs)'] * 3 + ['(in-lbs)'] * 3
    formats = ['{:.0f}'] * 9
    make_table(sec, "Base Plate Attachment Forces", header, units, data, col_formats=formats,
               rows_to_highlight=governing_cxn_idx)
    sec.append(NewLine())


def base_straps(item, sec, sec_title, sub_title, plots_dict):
    sec.append(NoEscape(
        r'\textit{Base Straps} represent hardware elements which connect the equipment unit to base plates or other'
        r' floor-anchorage hardware.'))

    sol = item.governing_solutions['base_anchor_tension']['sol']
    item.update_element_resultants(sol)

    Tn = item.base_straps[0].brace_capacity
    Rn_eq = item.base_straps[0].capacity_to_equipment
    Rn_backing = item.base_straps[0].capacity_to_backing
    check_brace = isinstance(Tn, (int, float))
    check_to_eq = isinstance(Rn_eq, (int, float))
    check_to_backing = isinstance(Rn_backing, (int, float))

    if not any([check_brace, check_to_eq, check_to_backing]):
        sec.append(
            "By inspection, strap elements are determined not to be the governing component of the load path.")
        return

    strap = max(item.base_straps, key=lambda x: x.brace_force)
    Tu = strap.brace_force
    dcr = strap.tension_dcr
    sec.append("The strap force corresponding to maximum base anchor tension is:")
    with sec.create(Math(inline=False)) as m:
        m.append(NoEscape(rf'T_u = {Tu:.2f} \quad \text{{ lbs}}'))

    if strap.capacity_method == 'ASD':
        sec.append("The equipment model was analyzed under ultimate (LRFD) level forces. "
                   "Base strap capacities are tabulated as allowable (ASD) capacities. Maximum strap tension and "
                   "is converted by multiplying by the ratio of ASD-factored to LRFD-factored "
                   "lateral loads:")
        T_asd = Tu * item.asd_lrfd_ratio
        with sec.create(Flalign()) as fl:
            fl.append(NoEscape(
                rf'T_{{ASD}} &= T_{{LRFD}}\left(\frac{{ F_{{h,ASD}} }}{{ F_{{h,LRFD}} }}\right) &&= {Tu:.2f}\left(\frac{{ {item.Fah:.2f} }}{{ {item.Fuh:.2f} }}\right)&&=({item.Fah / item.Fuh:0.2f}){Tu:.2f} &&={T_asd:.2f} \text{{ (lbs)}}\\'))
        sec.append(
            "Capacities for bracket elements are provided by manufacturer data or pre-tabulated by the engineer.")
        if check_brace:
            subheader(sec, "Strap Capacity Check")
            with sec.create(Flalign()) as align:
                align.append(
                    rf'&DCR &&=\frac{{T_{{ASD}} }}{{T_n/\Omega}} &&=\frac{{({T_asd:.2f})}}{{({Tn})}} &&= {dcr:.2f}\\')
        if check_to_eq:
            subheader(sec, "Strap Connection to Equipment")
            with sec.create(Flalign()) as align:
                align.append(
                    rf'&DCR &&=\frac{{T_{{ASD}}}}{{R_n/\Omega }} &&=\frac{{({Tu:.2f})}}{{({Rn_eq})}} &&= {Tu / Rn_eq:.2f}\\')
        if check_to_backing:
            subheader(sec, "Strap Connection to Base Plates")
            with sec.create(Flalign()) as align:
                align.append(
                    rf'&DCR &&=\frac{{T_{{ASD}}}}{{R_n/\Omega }} &&=\frac{{({Tu:.2f})}}{{({Rn_backing})}} &&= {Tu / Rn_backing:.2f}\\')

    else:  # LRFD Method

        sec.append(
            "Capacities for bracket elements are provided by manufacturer data or pre-tabulated by the engineer.")
        if check_brace:
            subheader(sec, "Strap Capacity Check")
            with sec.create(Flalign()) as align:
                align.append(rf'&DCR &&=\frac{{T_u}}/{{\phi T_n}} &&=\frac{{({Tu:.2f})}}{{({Tn})}} &&= {Tu / Tn:.2f}\\')
        if check_to_eq:
            subheader(sec, "Strap Connection to Equipment")
            with sec.create(Flalign()) as align:
                align.append(rf'&DCR &&=T_u/{{\phi R_n}} &&=\frac{{({Tu:.2f})}}{{({Rn_eq})}} &&= {Tu / Rn_eq:.2f}\\')
        if check_to_backing:
            subheader(sec, "Strap Connection to Base Plates")
            with sec.create(Flalign()) as align:
                align.append(
                    rf'&DCR &&=T_u/{{\phi R_n}} &&=\frac{{({Tu:.2f})}}{{({Rn_backing})}} &&= {Tu / Rn_backing:.2f}\\')


def wall_brackets(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict):
    # sec.append(NoEscape(
    #     r'\textit{Wall brackets} represent the hardware or discrete points of attachment connecting the'
    #     r' equipment unit to the wall anchors. Brackets may impart loads to a single anchor, '
    #     r'or to multiple anchors through a \textit{backing element}. Bracket forces are determined '
    #     r'from the analytical model. Bracket forces are distributed to backing and wall anchors using '
    #     r'the elastic bolt group method.'))
    sec.append(NoEscape(
        r'\textit{Wall brackets} represent the hardware or discrete points of attachment connecting the'
        r' equipment unit to the wall anchors. Brackets may impart loads to a single anchor, '
        r'or to multiple anchors through a \textit{backing element}. Bracket forces are determined '
        r'from the analytical model. Bracket forces are distributed to backing and wall anchors using '
        r'the elastic bolt group method.'))

def wall_bracket_demands(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict):
    run = model_record.analysis_runs[model_record.governing_run]
    brackets = run.results.wall_brackets

    # Bracket Forces Plots
    title = f'{run.hardware_selection.bracket_id} Forces vs. Direction of Loading'
    # matrix_n = item.wall_bracket_forces[:, :, 0]
    # matrix_p = item.wall_bracket_forces[:, :, 1]
    # matrix_z = item.wall_bracket_forces[:, :, 2]
    # fig, width = plots._forces_vs_theta(item.theta_z,
    #                                     [matrix_n, matrix_p, matrix_z],
    #                                     [item.governing_solutions['wall_bracket_tension'][
    #                                         'theta_z'],
    #                                     item.governing_solutions['wall_bracket_shear'][
    #                                         'theta_z'],
    #                                     item.governing_solutions['wall_bracket_shear'][
    #                                         'theta_z']],
    #                                     [r'Normal, $N$ (lbs)', r'In-Plane Shear, $V_p$ (lbs)',
    #                                     r'Vert. Shear, $V_z$ (lbs)'],
    #                                     ['N', 'V_p', 'V_z'])
    # file = plots.plt_save()
    fig_name = 'bracket_forces'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file, title=title)

    # Displaced Shape Figure
    # fig, width = plots._displaced_shape(item, sol, theta_z)
    # file = plots.plt_save()

    fig_name = 'displaced_shape'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    with sec.create(MiniPage(width=f'{width}in',pos='t',align='r')) as mini:
        make_figure(mini, width, file)

        # Bracket Forces Table
        sec.append(NewLine())
        sec.append(NoEscape(r'\smallskip'))
        title = 'Bracket Forces at Governing Load Angle'
        header = [NoEscape('$x$'), NoEscape('$y$'), NoEscape('$z$'),
                  NoEscape('$N$'), NoEscape('$V$'), NoEscape('$Z$')]
        units = ['(in)', '(in)', '(in)', '(lbs)', '(lbs)', '(lbs)']
        data = [[bracket.xyz_equipment[0], bracket.xyz_equipment[1],
                 bracket.xyz_equipment[2], bracket.bracket_forces["fn"],
                 bracket.bracket_forces['fp'],
                 bracket.bracket_forces['fz']] for bracket in brackets]
        highlight_idx = np.argmax([bracket.bracket_forces['fn'] for bracket in brackets])
        formats = ['{:.2f}', '{:.2f}', '{:.2f}', '{:.0f}', '{:.0f}', '{:.0f}']
        alignment = 'cccccc'
        make_table(mini, title, header, units, data, col_formats=formats, alignment=alignment, width=r'3.5in',
                   rows_to_highlight=highlight_idx)


def wall_bracket_checks(model_record: ModelRecord,
        sec: Union[Section, Subsection, Subsubsection],
        sec_title: str,
        sub_title: str,
        plots_dict: dict):
    run = model_record.analysis_runs[model_record.governing_run]
    brackets = run.results.wall_brackets
    br_res, br_idx = get_governing_result(brackets)
    bracket = model_record.model.elements.wall_brackets[br_idx]

    Tn = bracket.bracket_props.bracket_capacity
    Rn_eq = bracket.bracket_props.capacity_to_equipment
    Rn_backing = bracket.bracket_props.capacity_to_backing
    check_bracket = isinstance(Tn, (int, float))
    check_to_eq = isinstance(Rn_eq, (int, float))
    check_to_backing = isinstance(Rn_backing, (int, float))

    if not any([check_bracket, check_to_eq, check_to_backing]):
        sec.append(
            "By inspection, bracket elements are determined not to be the governing component of the load path.")
        return
    Tu = bracket.fn.max()
    
    sec.append(NoEscape(rf"The maximum bracket tension force for any angle of loading is: $T_u = {Tu:.2f}$\\"))
    # with sec.create(Math(inline=False)) as m:
    #     m.append(NoEscape(rf'T_u = {Tu:.2f}'))
    sec.append(
        "Capacities for bracket elements are taken from manufacturer data or pre-tabulated by the engineer.")
    sec.append(NewLine())
    if check_bracket:
        sec.append(NoEscape(rf'\textbf{{\textit{{{run.hardware_selection.bracket_id} Capacity Check}}}}'))
        with sec.create(Flalign()) as align:
            align.append(rf'&DCR &&=T_u/{{\phi T_n}} &&=\frac{{({Tu:.2f})}}{{({Tn})}} &&= {Tu / Tn:.2f}')
    if check_to_eq:
        sec.append(NoEscape(rf'\textbf{{\textit{{{run.hardware_selection.bracket_id} Connection to Equipment}}}}'))
        with sec.create(Flalign()) as align:
            align.append(rf'&DCR &&=T_u/{{\phi R_n}} &&=\frac{{({Tu:.2f})}}{{({Rn_eq})}} &&= {Tu / Rn_eq:.2f}')
    if check_to_backing:
        sec.append(NoEscape(rf'\textbf{{\textit{{{run.hardware_selection.bracket_id} Connection to Wall Backing}}}}'))
        with sec.create(Flalign()) as align:
            align.append(
                rf'&DCR &&=T_u/{{\phi R_n}} &&=\frac{{({Tu:.2f})}}{{({Rn_backing})}} &&= {Tu / Rn_backing:.2f}')


def wall_anchor_demands(model_record: ModelRecord,
        sec: Union[Section, Subsection, Subsubsection],
        sec_title: str,
        sub_title: str,
        plots_dict: dict,
        governing_backing_idx):
    run = model_record.analysis_runs[model_record.governing_run]
    wall_anchors = run.results.wall_anchors
    wa_res, wa_idx = get_governing_result(wall_anchors)
    wa = model_record.model.elements.wall_anchors[wa_idx]
    sec.append("An anchor group is taken as all anchors within a single backing element. "
               "Forces to wall anchors are determined from the combined effect of all brackets "
               r"acting on the backing element. The backing element with wall anchors and bracket forces is shown below.")
    sec.append(NewLine())

    sec.append(NoEscape(
        'Equivalent centroid forces (normal, shear and moments) on the backing group are computed '
        'from bracket forces. Let $N_{br}$, $V_{x,br}$,$V_{y,br}$, $x_{br}$, and $y_{br}$ be the bracket force normal to the wall, shear forces, and the $x$- and $y$-coordinates of the bracket relative to the anchor group centroid:'))
    with sec.create(Flalign()) as align:
        align.append(
            r'N &= \sum{N_{br}}, \quad V_x = \sum{V_{x,br}}, \quad V_y = \sum{V_{x,br}}, \quad M_x = \sum{N_{br}x_{br}}, \quad M_y = \sum{-N_{br}y_{br}}, \quad T = \sum{V_{y,br}x_{br}-V_{x,br}y_{br}}&')

    sec.append(NoEscape(
        'Tension and shear forces in anchors, $T_a$ and $V_a$, are computed by the following relationships:'))
    with sec.create(Flalign()) as align:
        align.append(
            r'T_a &=\frac{N}{n_a} +\left(\frac{M_xI_{yy}-M_yI_{xy}}{I_{xx}I_{yy}-I_{xy}^2}\right)y+\left(\frac{M_yI_{xx}-M_xI_{xy}}{I_{xx}I_{yy}-I_{xy}^2}\right)x&')
        align.append(
            r'V_a &= \sqrt{V_{ax}^2 + V_{ay}^2}, & V_{ax} &= \frac{V_x}{n_a} - \frac{M_zy}{I_p}, & V_{ay} &= \frac{V_y}{n_a} + \frac{M_zx}{I_p}&')

    # Wall Anchor Forces Plot
    title = 'Wall Fastener Forces vs. Direction of Loading'

    fig_name = 'anchor_forces'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file, title=title)

    # Displaced Shape Figure
    fig_name = 'displaced_shape'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file)

    sec.append(NewLine())
    sec.append(' ')
    sec.append(NoEscape(r'\smallskip'))

    fig_name = 'backing'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file, title='Free-body Diagram of Backing Element')

    sec.append(NoEscape(r'\hfill'))

    # Anchor Force Table
    header = [NoEscape('Normal, $N$'), NoEscape('Shear, $V_x$'), NoEscape('Shear, $V_y$')]
    units = ['(lbs)', '(lbs)', '(lbs)']
    formats = ['{:.0f}', '{:.0f}', '{:.0f}']
    # data = np.column_stack((governing_backing.anchor_forces[:, 0],
    #                         np.linalg.norm(governing_backing.anchor_forces[:, 1:], axis=1)))

    data = wa_res.forces[:,:,wa_res.governing_theta_idx]
    make_table(sec, "Governing Wall Fastener Forces", header, units, data, alignment='ccc', col_formats=formats,
               width='3in')


def bracket_connection_demands(model_record: ModelRecord,
        sec: Union[Section, Subsection, Subsubsection],
        sec_title: str,
        sub_title: str,
        plots_dict: dict):
    # Attachment Forces Table
    run = model_record.analysis_runs[model_record.governing_run]
    brackets = run.results.wall_brackets
    gov_bracket, gov_bracket_idx = get_governing_result(brackets)


    data = [bracket.reactions_equipment[:,gov_bracket.governing_theta_idx] for bracket in brackets]
    sec.append(
        "The table below summarizes the internal reaction forces at the attachment points "
        "for wall bracket elements. Connection of the brackets to the equipment unit "
        "(through welds, fasteners, etc.) must be designed to accommodate these forces.")

    sec.append(NewLine())
    header = [NoEscape('$R_x$'),
              NoEscape('$R_y$'),
              NoEscape('$R_z$'),
              NoEscape('$M_x$'),
              NoEscape('$M_y$'),
              NoEscape('$M_z$')]
    units = ['(lbs)'] * 3 + ['(in-lbs)'] * 3
    formats = ['{:.0f}'] * 6
    make_table(sec, "Bracket Attachment Forces", header, units, data, col_formats=formats, rows_to_highlight=[gov_bracket_idx])
    sec.append(NewLine())


def sms_connection_demands(
        model_record: ModelRecord,
        sec: Union[Section, Subsection, Subsubsection],
        sec_title: str,
        sub_title: str,
        plots_dict: dict,
        cxn_obj,
        cxn_res,
        anchors_obj: sms.SMSAnchors,
        results: sms.SMSResults,
        connection_type):

    theta_idx = results.governing_theta_idx
    """section taking nodal demands and applying them to sms group"""
    type_to_text = {'base': 'Base plates',
                    'bracket': 'Bracket'}

    # sec.append(NoEscape(r'\bigskip'))
    # sec.append(NewLine())
    sec.append(NoEscape(
        f'{type_to_text[connection_type]} are attached to the equipment using sheet-metal-screws. '
        'Centroid forces (normal, shear and moments) on the connection screws group are computed '
        r'from nodal forces at the connection of the floor plate element to equipment model.\\'))
    sec.append(NoEscape(r'\smallskip'))
    sec.append(NoEscape(
        'Given connection forces $N$, $V_x$, $M_x$, $M_y$, $V_y$, $T$, the tension and shear forces in anchors, $T_a$ and $V_a$, are computed by the following relationships:'))

    with sec.create(Flalign()) as align:
        align.append(
            r'T_a &=\frac{N}{n_a} +\left(\frac{M_xI_{yy}-M_yI_{xy}}{I_{xx}I_{yy}-I_{xy}^2}\right)y+\left(\frac{M_yI_{xx}-M_xI_{xy}}{I_{xx}I_{yy}-I_{xy}^2}\right)x,&')
        align.append(
            r'V_a &= \sqrt{V_{ax}^2 + V_{ay}^2}, & V_{ax} &= \frac{V_x}{n_a} - \frac{M_zy}{I_p}, & V_{ay} &= \frac{V_y}{n_a} + \frac{M_zx}{I_p}&\\')


    fig_name = 'sms'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file, title='Connection F.B.D.')

    sec.append(NoEscape(r'\hfill'))

    # Anchor Force Table
    header = [NoEscape('Normal, $T$'), NoEscape('Shear, $V_x$'), NoEscape('Shear, $V_y$')]
    units = ['(lbs)', '(lbs)', '(lbs)']
    formats = ['{:.0f}', '{:.0f}', '{:.0f}']
    data = np.column_stack((results.tension_demand[:, theta_idx],
                            results.shear_x_demand[:, theta_idx],
                            results.shear_y_demand[:,theta_idx]))
    make_table(sec, "Fastener Demands for Governing Connection", header, units, data,
               alignment='ccc', col_formats=formats, width='3in')


def sms_checks(
        model_record: ModelRecord,
        sec: Union[Section, Subsection, Subsubsection],
        sec_title: str,
        sub_title: str,
        plots_dict: dict,
        anchors_obj: sms.SMSAnchors,
        results: sms.SMSResults):

    # Extract Parameters
    a = anchors_obj
    theta_idx = results.governing_theta_idx
    a_idx = results.governing_anchor_idx

    condition_labels = {SMSCondition.METAL_ON_METAL: {'Label':'Steel-to-Steel Connection',
                                                      'Table': 'Table 1'},
                        SMSCondition.GYP_1_LAYER: {'Label': 'Single Layer Gyp',
                                                   'Table': 'Table 2'},
                        SMSCondition.GYP_2_LAYERS: {'Label': 'Two Layer Gyp',
                                                   'Table': 'Table 3'},
                        SMSCondition.PRYING: {'Label': 'Prying',
                                                   'Table': 'Table 4'}
                        }

    #Todo: test what happens with non-permitted combinations of size, gauge and fy

    sec.append(NoEscape(
        "Allowable strengths of sheet metal screws in shear and tension are based upon tabulated values "
        "provided by the California Department of Healcare Access and Information (HCAI) OPD-0001-13: "
        r"\textit{Standard Partition Wall Details.}"))

    sec.append(
        NoEscape(r'SMS capacities are based on  screw size, base material, and attachment condition.'))

    sec.append(NoEscape(r'\bigskip'))
    sec.append(NewLine())

    with sec.create(MiniPage(width = '4in',pos='t', align='l')) as mini:
        mini.append(NoEscape(r'\begin{footnotesize}'))
        with mini.create(Tabular('ll')) as table:
            table.add_hline()
            table.add_row([NoEscape(r'\rowcolor{lightgray} Properties'), ''])
            table.add_hline()
            table.add_row(['Fastener Size', a.screw_size])
            table.add_hline()
            table.add_row([NoEscape(r'Steel $F_y$ (ksi)'), f'{a.props.fy:.0f}'])
            table.add_hline()
            table.add_row(['Steel Gauge', f'{a.props.gauge:.0f}'])
            table.add_hline()
            table.add_row(['Shear X Condition', condition_labels[a.props.condition_x]['Label']])
            table.add_row(['OPD Table', condition_labels[a.props.condition_x]['Table']])
            table.add_hline()
            table.add_row(['Shear Y Condition', condition_labels[a.props.condition_y]['Label']])
            table.add_row(['OPD Table', condition_labels[a.props.condition_y]['Table']])
            table.add_hline()
        mini.append(NoEscape(r'\end{footnotesize}'))

    sec.append(NoEscape(r'\hfill'))

    with sec.create(MiniPage(width='2.25in',pos='t',align='r')) as mini:
        if not True: # permissible:
            sec.append(NoEscape(
                r'\textcolor{red}{\textbf{The specified fastener size or material grade is not permitted! No capacity is reported.}}'))
        else:
            mini.append(NoEscape(r'\begin{footnotesize}'))
            #
            with mini.create(Tabular('ll')) as table:
                table.add_hline()
                # Insert row color before the row
                table.append(NoEscape(r'\rowcolor{lightgray}'))
                table.add_row([
                    MultiColumn(2, align='l', data=NoEscape(r'OPD-0001 Tabulated Capacities'))
                ])

                table.add_hline()
                table.add_row([NoEscape(r'$T_{all}$ (lbs)'), f'{results.tension_capacity:.0f}'])
                table.add_row([NoEscape(r'$V_{x,all}$ (lbs)'), f'{results.shear_x_capacity:.0f}'])
                table.add_row([NoEscape(r'$V_{y,all}$ (lbs)'), f'{results.shear_y_capacity:.0f}'])
                table.add_hline()
            mini.append(NoEscape(r'\end{footnotesize}'))
    sec.append(NoEscape(r'\bigskip'))
    sec.append(NewLine())

        # if not permissible:
        #     ss.append(NoEscape(
        #         r'\textcolor{red}{\textbf{The specified fastener size or material grade is not permitted! No capacity is reported.}}'))
        # else:
        #     ss.append('The tabulated capacities for shear and tension are:')
        #
        #     with ss.create(Flalign()) as fl:
        #         fl.append(rf'T_{{all}} &= {T_all:.0f} (lbs) &\\')
        #         fl.append(rf'V_{{x,all}} &= {Vx_all:.0f} (lbs) &\\')
        #         fl.append(rf'V_{{y,all}} &= {Vy_all:.0f} (lbs) &')

    t = results.tension_demand[a_idx, theta_idx]
    t_cap = results.tension_capacity
    t_unity = results.tension_unity[a_idx, theta_idx]

    vx = results.shear_x_demand[a_idx, theta_idx]
    vx_cap = results.shear_x_capacity
    vx_unity = results.shear_x_unity[a_idx, theta_idx]

    vy = results.shear_y_demand[a_idx, theta_idx]
    vy_cap = results.shear_y_capacity
    vy_unity = results.shear_y_unity[a_idx, theta_idx]

    v_unity = results.shear_unity[a_idx, theta_idx]

    ok_t = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if t_unity <= 1 else r'\textcolor{red}{\textbf{\textsf{NG}}}'
    ok_vx = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if vx_unity <= 1 else r'\textcolor{red}{\textbf{\textsf{NG}}}'
    ok_vy = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if vy_unity <= 1 else r'\textcolor{red}{\textbf{\textsf{NG}}}'
    ok_v = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if v_unity <= 1 else r'\textcolor{red}{\textbf{\textsf{NG}}}'
    ok = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if results.ok else r'\textcolor{red}{\textbf{\textsf{NG}}}'
    sec.append('Three conditions of loading are considerd; maximum tension temand, maximum shear demand (in both x- and y-directions) and loading resulting in maximum combined :')
    with sec.create(Flalign()) as fl:
        fl.append(
            rf'DCR_T &= T_{{ASD}} / T_{{all}} &&={t:.0f}/{t_cap:.0f} &&={t_unity:.2f} &{ok_t}\\')
        fl.append(
            rf'DCR_{{Vx}} &= V_{{x,ASD}} / V_{{x,all}} &&={vx:.0f}/{vx_cap:.0f} &&={vx_unity:.2f} &{ok_vx}\\')
        fl.append(
            rf'DCR_{{Vy}} &= V_{{y,ASD}} / V_{{y,all}} &&={vy:.0f}/{vy_cap:.0f} &&={vy_unity:.2f} &{ok_vy}\\')
        fl.append(
            rf'DCR_V &= \sqrt{{DCR_{{Vx}}^2 + DCR_{{Vx}}^2}} &&=\sqrt{{ {vx_unity:.2f}^2 +{vy_unity:.2f}^2}}&&={v_unity:.2f} &{ok_v}\\')
        fl.append(
            rf'DCR_T &= DCR_T + DCR_V &&= {t_unity:.2f} + {v_unity:.2f} &&={results.unity:.2f} &{ok}\\')

def wood_fastener_checks(item, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj
    sec.append(
        NoEscape(r'Wood fastener capacities are calculated below.'))

    sec.append(NoEscape(r'\bigskip'))
    sec.append(NewLine())

    with sec.create(MiniPage(width='4in', pos='t', align='l')) as mini:
        mini.append(NoEscape(r'\begin{footnotesize}'))
        with mini.create(Tabular('ll')) as table:
            table.add_hline()
            table.add_row([NoEscape(r'\rowcolor{lightgray} Properties'), ''])
            table.add_hline()
            table.add_row(['Fastener ID', a.fastener_id])
            table.add_row(['Fastener Type', a.fastener_type])
            table.add_row(['Diameter', f'{a.D:.2f}'])
            table.add_hline()
        mini.append(NoEscape(r'\end{footnotesize}'))



    # Reference Withdrawal Values
    # sec.append(NoEscape(r"\textit{Reference Withdrawal Design Values}"))
    sec.append(NoEscape(r'\smallskip'))
    sec.append(NewLine())
    # subheader(sec, 'Reference Withdrawal Design Values')
    sec.append(NoEscape(r'\textit{\textbf{Reference Withdrawal Design Values}}'))
    sec.append(NewLine())
    math_lines = [[]]
    if a.fastener_type == 'Lag Screw':
        math_lines = [['W=1800G^{(3/2)}D^{(3/4)}',rf'={a.W:.0f} \text{{ lbs}}', 'NDS \S12.2-1']]
    if a.fastener_type == 'Wood Screw':
        math_lines = [['W=2850G^{2}D',rf'={a.W:.0f} \text{{ lb/in}}', 'NDS \S12.2-2']]
    math_alignment_longtable(sec, math_lines, width='6.5in')

    # Reference Lateral Design Values
    # sec.append(NewLine())
    sec.append(NoEscape(r'\textit{\textbf{\noindent{Reference Lateral Design Values}}}'))
    sec.append(NewLine())
    sec.append(NoEscape(r'Lateral design values are based on the NDS and \textit{AWC Technical Report 12}. (See reference equations in TR12 Table 1-1.)\\'))
    sec.append(NoEscape(r'\bigskip'))


    with sec.create(MiniPage(width='2in', pos='t', align='l')) as mini:
        mini.append(NoEscape(r'\begin{footnotesize}'))
        with mini.create(Tabular('ll')) as table:
            table.add_hline()
            table.add_row([NoEscape(r'\rowcolor{lightgray} Input Parameters'), ''])
            table.add_hline()
            table.add_row([NoEscape(r'$L_m$ (in)'), f'{a.p:.2f}'])
            table.add_row([NoEscape(r'$L_s$ (in)'), f'{a.t_steel:.2f}'])
            table.add_row([NoEscape(r'$q_m = F_{em}D$ (lb/in)'), f'{a.Fem*a.D:.0f}'])
            table.add_row([NoEscape(r'$q_s = F_{es}D$ (lb/in)'), f'{a.Fes * a.D:.0f}'])
            table.add_row([NoEscape(r'$M_m = M_s =  F_{yb}D^3/6$ (lb-in)'), f'{a.Fyb * (a.D**3/6):.0f}'])
            table.add_row([NoEscape(r'$K_{\theta} = 1+0.25(\theta/90)$'), f'{a.K_theta:.2f}'])
            if a.D<0.25:
                table.add_row([NoEscape(r'$K_D$'), f'{a.K_D:.2f}'])
            table.add_row([NoEscape(r'Gap, $g$ (in)'),f'{a.g:.2f}'])
            table.add_hline()

        mini.append(NoEscape(r'\end{footnotesize}'))

    sec.append(NoEscape(r'\hfill'))

    with sec.create(MiniPage(width='4in', pos='t', align='l')) as mini:
        mini.append(NoEscape(r'\begin{footnotesize}'))
        with mini.create(Tabular('lcc')) as table:
            table.add_hline()
            table.add_row([NoEscape(r'\rowcolor{lightgray} Yield Mode'), NoEscape(r'Reduction Term, $R_d$'), 'NDS Table 12.3.1B'])
            table.add_hline()
            if a.D<0.25:
                table.add_row([NoEscape(r'I\textsubscript{m},I\textsubscript{s},II,III\textsubscript{m},III\textsubscript{s},IV'),
                               NoEscape(r'$K_D$'),
                               f'{a.Rd[0]:.2f}'])
            else:
                table.add_row([NoEscape(r'I\textsubscript{m}'),
                               MultiRow(2, data=NoEscape(r'$4K_{\theta}$')),
                               MultiRow(2, data=f'{a.Rd[0]:.2f}')]),
                table.add_row([NoEscape(r'I\textsubscript{s}'),'', ''])
                table.add_row([NoEscape(r'II'),NoEscape(r'$3.6K_{\theta}$'),f'{a.Rd[2]:.2f}'])
                table.add_row([NoEscape(r'III\textsubscript{m}'),
                               MultiRow(2, data=NoEscape(r'$3.2K_{\theta}$')),
                               MultiRow(2, data=f'{a.Rd[3]:.2f}')])
                table.add_row([NoEscape(r'III\textsubscript{s}'),'',''])
                table.add_row([NoEscape(r'IV'),'',''])
            table.add_hline()

        mini.append(NoEscape(r'\end{footnotesize}'))

    sec.append(NoEscape(r'\bigskip'))
    sec.append(NewLine())

    with sec.create(MiniPage(width='6.5in', pos='t', align='l')) as mini:
        mini.append(NoEscape(r'\begin{footnotesize}'))
        with mini.create(Tabular('lllllr')) as table:
            table.add_hline()
            table.add_row([NoEscape(r'\rowcolor{lightgray} Yield Mode'),'Equation' ,'','','',NoEscape('$Z$ (lb/in)')])
            table.add_hline()
            table.add_row([NoEscape(r'I\textsubscript{m}'),
                           NoEscape(r'$q_mL_m/R_d$'), '','','',
                           f'{a.yield_modes["Im"]:.0f}'])
            table.add_row([NoEscape(r'I\textsubscript{s}'),
                           NoEscape(r'$q_sL_s/R_d$'), '', '', '',
                           f'{a.yield_modes["Is"]:.0f}'])
            table.add_row([NoEscape(r'II'),
                           MultiRow(4, data=NoEscape(r'$\frac{-B+\sqrt{B^2-4AC}}{2AR_d}$')),
                           NoEscape(r'$A=\frac{1}{4q_s}+\frac{1}{4q_m}$'),
                           NoEscape(r'$B=\frac{L_s}{2}+g+\frac{L_m}{2}$'),
                           NoEscape(r'$C=-\frac{q_sL^2_s}{4}-\frac{q_mL_m^2}{4}$'),
                           f'{a.yield_modes["II"]:.0f}'])
            table.add_row([NoEscape(r'III\textsubscript{m}'),
                           '',
                           NoEscape(r'$A=\frac{1}{2q_s}+\frac{1}{4q_m}$'),
                           NoEscape(r'$B=g+\frac{L_m}{2}$'),
                           NoEscape(r'$C=-M_s-\frac{q_mL_m^2}{4}$'),
                           f'{a.yield_modes["IIIm"]:.0f}'])
            table.add_row([NoEscape(r'III\textsubscript{s}'),
                           '',
                           NoEscape(r'$A=\frac{1}{4q_s}+\frac{1}{2q_m}$'),
                           NoEscape(r'$B=\frac{L_s}{2}+g$'),
                           NoEscape(r'$C=-\frac{q_sL^2_s}{4}-M_m$'),
                           f'{a.yield_modes["IIIs"]:.0f}'])
            table.add_row([NoEscape(r'IV'),
                           '',
                           NoEscape(r'$A=\frac{1}{2q_s}+\frac{1}{2q_m}$'),
                           NoEscape(r'$B=g$'),
                           NoEscape(r'$C=-M_s-M_m$'),
                           f'{a.yield_modes["IV"]:.0f}'])
            table.add_hline()

        mini.append(NoEscape(r'\end{footnotesize}'))

    sec.append(NewLine())

    # Adjustment Factors
    subheader(sec,"Adjustment Factors")
    sec.append(NoEscape(r"{\raggedright[COMMING SOON...]}"))

    # Adjusted Capacity Checks
    subheader(sec, "Combined Capacity Check")

    sec.append(NewLine())
    sec.append(r"The tension and shear resulting in the largest combined loading DCR are given below:")
    math_lines = [[f"N={a.N:.2f}",f"V_x={a.Vx:.2f}",f"V_y={a.Vy:.2f}",""]]
    math_alignment_longtable(sec,math_lines)

    sec.append(r"The adjusted capacities are:")
    math_lines = [['W^\prime=WC_MC_tC_{eg}K_f\phi',
                   f'=({a.W:.2f})({a.C_M:.2f})({a.C_t:.2f})({a.C_eg:.2f})({a.Kf:.2f})({a.phi:.2f})',
                   f'={a.W_prime:.2f}'],
                  ['Z^\prime=ZC_MC_tC_gC_{\delta}C_{eg}C_{di}C_{tn}K_f\phi',
                   f'=({a.Z:.2f})({a.C_M:.2f})({a.C_t:.2f})({a.C_g:.2f})({a.C_delta:.2f})({a.C_eg:.2f})({a.C_di:.2f})({a.C_tn:.2f})({a.Kf:.2f})({a.phi:.2f})',
                   f'={a.Z_prime:.2f}']
                  ]  #todo, add time_factor once you can see what the variable is called in NDS
    math_alignment_longtable(sec,math_lines)

    sec.append(NoEscape(r"The combined loading utilization is:\\"))
    sec.append(NoEscape(r'\begin{math}'))
    sec.append(NoEscape(r'Z^\prime_\alpha = \frac{W^\prime pZ^\prime}{W^\prime p\cos^2{\alpha}+Z^\prime \sin^2{\alpha}}='+rf'{a.z_alpha_prime:.0f}\\'))
    sec.append(NoEscape(r'\text{DCR}=V/Z^\prime_\alpha =' +rf'{a.V:.0f}/{a.z_alpha_prime:.0f}={a.DCR:.2f}'))
    sec.append(NoEscape(r'\end{math}'))

def concrete_summary_spacing_only(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict,
                          anchor_obj: conc.ConcreteAnchors, results: conc.ConcreteAnchorResults):
    with sec.create(MiniPage(width=r"3.75in", pos='t')) as mini:
        _concrete_input_parameters(mini,anchor_obj,results)

    _spacing_checks(model_record, sec, sec_title, sub_title, plots_dict, anchor_obj, results)


def concrete_summary_full(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict,
                          anchor_obj: conc.ConcreteAnchors, results: conc.ConcreteAnchorResults):
    a = anchor_obj
    tg_idx = results.governing_tension_group
    sg_idx = results.governing_shear_group
    idx_theta = results.governing_theta_idx
    idx_anchor = results.governing_anchor_idx

    sec.append('''Anchor capacity is evaluated according to the failure mode requirements of ACI 318-19. 
    Loads and details are given below. The anchor spacing and edge distances are checked against manufacturer requirements. ''')


    tg = results.tension_groups[tg_idx]

    sec.append(NoEscape(rf'''The governing condition consists of a tension breakout group of ({tg.n_anchor:1}) 
    anchor(s). '''))

    if results.shear_groups:
        sg = results.shear_groups[sg_idx]
        sec.append(NoEscape(rf'''All possible shear breakout groups were evaluated according to ACI 318-19, \S17.7.2.1.1.
        The governing shear breakout group of ({sg.n_anchor:1}) anchor(s). '''))
    else:
        sec.append(NoEscape(rf'''Shear breakout does not govern due to edge large distances to anchor(s). '''))
    # Anchorage Condition Table

    subheader(sec, 'Anchor Data')

    with sec.create(MiniPage(width=NoEscape(r"3.75in"), pos='t', align='l')) as mini:
        # Input Data Table
        _concrete_input_parameters(mini, anchor_obj, results)

    sec.append(NoEscape(r'\hfill'))
    with sec.create(MiniPage(width=NoEscape(r"2.5in"), pos='t', align='c')) as mini:
        mini.append('Table Here')
        # Anchors Table
        # header = [NoEscape(r'\#'),
        #     NoEscape(r'$x$'),
        #           NoEscape(r'$xy$'),
        #           NoEscape(r'$T_u$'),
        #           NoEscape(r'$V_{ux}$'),
        #           NoEscape(r'$V_{uy}$')]
        # units = ['', '(in)', '(in)', '(lbs)', '(lbs)', '(lbs)']
        # data = []
        # for i, (x, y) in enumerate(a.geo_props.xy_anchors[tg.anchor_indices]):
        #
        #     data.append([a.group_idx[i], x, y, *a.max_group_forces["tension"][i, :]])
        # formats = ['{:.0f}', '{:.2f}', '{:.2f}', '{:.0f}', '{:.0f}', '{:.0f}']
        # make_table(mini, None, header, units, data, alignment='lccccc',
        #            col_formats=formats, use_minipage=False,add_index=False)

        # mini.append(NewLine())
        fig_name = 'diagram'
        width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
        make_figure(mini, width, file, use_minipage=False)





    # Anchor Spacing and Edge Distance Checks
    _spacing_checks(model_record, sec, sec_title, sub_title, plots_dict, anchor_obj, results)

    # Limit States Table
    # limit state: (calc, Label, Mode, Capacity Attribute Name)
    tension_limits: List[Tuple[Any, str, str, str]] = [
        (results.steel_tension_calcs[tg_idx],"Steel Tensile Strength", "Tension", "Nsa"),
        (results.tension_breakout_calcs[tg_idx], "Concrete Tension Breakout", "Tension", "Ncb"),
        (results.anchor_pullout_calcs[tg_idx],"Anchor Pullout","Tension", "Np"),
        (results.side_face_blowout_calcs[tg_idx], "Side Face Blowout", "Tension", "Nsbg"),
        (results.bond_strength_calcs[tg_idx], "Bond Strength", "Tension", "Nag")]

    shear_limits: List[Tuple[Any, str, str, str]] = [
        (results.steel_shear_calcs[tg_idx], "Steel Shear Strength", "Shear", "Vsa"),
        (results.shear_pryout_calcs[tg_idx], "Shear Pryout", "Shear", "Vcp")]
    if results.shear_breakout_calcs:
        shear_limits.append((results.shear_breakout_calcs[sg_idx],"Concrete Shear Breakout", "Shear", "Vcb"))


    sec.append(NoEscape(r'\begin{samepage}'))
    sec.append(NoEscape(r'\begin{footnotesize}'))
    subheader(sec, 'Anchor Limit States')
    with sec.create(Tabularx('Xcrrrrrr')) as table:
        table.add_hline()
        tension_header = [NoEscape(r'\rowcolor{lightgray} Tension Limits'),
                  'Mode',
                  NoEscape('$N_u$'),
                  NoEscape('$N_n$'),
                  NoEscape(r'$\phi$'),
                  NoEscape(r'$\phi_{seismic}$'),
                  NoEscape(r'$\phi\phi_{seismic}N_n$'),
                  'Utilization']
        shear_header = [NoEscape(r'\rowcolor{lightgray} Shear Limits'),
                          'Mode',
                          NoEscape('$V_u$'),
                          NoEscape('$V_n$'),
                          NoEscape(r'$\phi$'),
                          NoEscape(r'$\phi_{seismic}$'),
                          NoEscape(r'$\phi\phi_{seismic}V_n$'),
                          'Utilization']
        units = [NoEscape(r'\rowcolor{lightgray}'), '', '(lbs)', '(lbs)', '', '', '(lbs)', '']



        table.add_row(tension_header)
        table.add_row(units)
        table.add_hline()
        def add_limit_rows(table,limits_list):
            for calc, label, mode, cap_par in limits_list:
                if calc is not None:
                    demand = calc.demand[idx_theta]
                    phi = calc.phi
                    phi_seismic = calc.phi_seismic
                    capacity = getattr(calc,cap_par)
                    if isinstance(capacity,np.ndarray):
                        capacity = capacity[idx_theta]
                    product = phi*phi_seismic*capacity
                    unity = calc.unities[:,idx_theta].max()
                    row =[label,
                           mode,
                           f'{demand:.0f}',
                           f'{capacity:.0f}',
                          f'{phi:.2f}',
                          f'{phi_seismic:.2f}',
                          f'{product:.0f}',
                          f'{unity:.2f}']
                    table.add_row(row)
                    table.add_hline()

        add_limit_rows(table, tension_limits)
        table.add_row(shear_header)
        table.add_hline()
        add_limit_rows(table, shear_limits)

    sec.append(NoEscape(r'\end{footnotesize}'))
    sec.append(NoEscape(r'\end{samepage}'))



    # Tension-Shear Interaciton
    sec.append(NoEscape(r'\bigskip'))
    sec.append(NewLine())

    with sec.create(MiniPage(width=r"3.75in", pos='t')) as mini:
        subheader_nobreak(mini,'Tension-Shear Interaction')
        mini.append(NoEscape(rf'''Tension and shear are combined using the interaction criteria provided in 
                            ACI318-19, R17.8. The resulting interaction expression is given below:'''))

        ok = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if results.unity <= 1 else r'\textcolor{red}{\textbf{\textsf{NG}}}'
        equality = r'\leq' if results.unity <= 1 else r'>'
        with mini.create(Math(inline=False)) as math:
            math.append(NoEscape(
                rf'\left(\frac{{N_u}}{{\phi N_n}}\right)^{{\frac{{5}}{{3}}}}+\left(\frac{{V_u}}{{\phi V_n}}\right)^{{\frac{{5}}{{3}}}} \\'))
            math.append(NoEscape(
                rf'= \left({results.tension_unity_by_anchor[idx_anchor,idx_theta]:.2f}\right)^{{\frac{{5}}{{3}}}}+\left({results.shear_unity_by_anchor[idx_anchor,idx_theta]:.2f}\right)^{{\frac{{5}}{{3}}}} = {results.unity:.2f}'))
            math.append(NoEscape(rf'{equality} 1 \quad \text{{{ok}}}'))

        # Pull-test Values
        if model_record.include_pull_test:
            # mini.append(NoEscape(r'\smallskip'))
            mini.append(NewLine())
            subheader(mini, 'Minimum Anchor Pull-Test Value')
            Tu_max = results.forces[:,0,:].max()
            with mini.create(Flalign(numbering=False, escape=False)) as align:
                align.append(rf'''&N_{{\text{{Test}} }}
                    &&=3T_u \geq 500 \text{{ lbs}}  
                    &&= \left(3\right)\left({Tu_max:.2f}\right) \geq 500
                    &&={max([3 * Tu_max, 500]):.0f} \text{{ lbs}}
                    &\quad \text{{CAC22 (\S6-11.3.2)}}''')


    sec.append(NoEscape(r'\hfill'))
    fig_name = 'interaction'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file)



def _concrete_input_parameters(container,
                               anchor_obj: conc.ConcreteAnchors,
                               results: conc.ConcreteAnchorResults):
    a = anchor_obj
    tg = results.tension_groups[results.governing_tension_group]
    container.append(NoEscape(r'\begin{footnotesize}'))
    with container.create(Tabularx('lX', pos='t')) as table:
        table.add_hline()
        header = [NoEscape(r'\rowcolor{lightgray} Input Parameters'), '']
        table.add_row(header)
        table.add_hline()
        table.add_row(['Condition', r"Cracked(Assumed)"])
        table.add_hline()
        table.add_row(['Member Profile', rf'{a.concrete_props.profile}'])
        table.add_hline()
        table.add_row(['Member Thickness (in)', rf'{a.concrete_props.t_slab:.1f}'])
        table.add_hline()
        table.add_row([NoEscape(r"$f'_c$ (psi)"), rf'{a.concrete_props.fc:.0f}'])
        table.add_hline()
        table.add_row([NoEscape(r"$\lambda_a$"), rf'{a.concrete_props.lw_factor_a:.2f}'])
        table.add_hline()
        table.add_row([NoEscape('Anchor Spacing (in), $s_{min}$'), rf'{a.geo_props.s_min:.2f}'])
        table.add_hline()
        # Use ca values for anchor group, or c values for equipment geometry (if not analysis was run)
        cx_neg = tg.cax_neg #if a.cax_neg is not None else a.cx_neg
        cx_pos = tg.cax_pos #if a.cax_pos is not None else a.cx_pos
        cy_neg = tg.cay_neg #if a.cay_neg is not None else a.cy_neg
        cy_pos = tg.cay_pos #if a.cay_pos is not None else a.cy_pos
        # Convert edge distances to text
        cax_neg_text = NoEscape(r'$\infty$') if np.isinf(cx_neg) else rf'{cx_neg:.2f}'
        cax_pos_text = NoEscape(r'$\infty$') if np.isinf(cx_pos) else rf'{cx_pos:.2f}'
        cay_neg_text = NoEscape(r'$\infty$') if np.isinf(cy_neg) else rf'{cy_neg:.2f}'
        cay_pos_text = NoEscape(r'$\infty$') if np.isinf(cy_pos) else rf'{cy_pos:.2f}'
        table.add_row(['Edge Distances (in)', ''])
        table.add_row([NoEscape(r'\hfill $c_{ax-}$'), cax_neg_text])
        table.add_row([NoEscape(r'\hfill $c_{ax+}$'),cax_pos_text])
        table.add_row([NoEscape(r'\hfill $c_{ay-}$'), cay_neg_text])
        table.add_row([NoEscape(r'\hfill $c_{ay+}$'), cay_pos_text])
        table.add_hline()
        table.add_row(['Anchor Type', rf'{a.anchor_props.info.anchor_id}'])
        table.add_hline()
        table.add_row(['ESR', rf'{a.anchor_props.info.esr:.0f}'])
        table.add_hline()
    container.append(NoEscape(r'\end{footnotesize}'))

def _spacing_checks(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict,
                    anchor_obj:conc.ConcreteAnchors,
                    anchor_results: conc.ConcreteAnchorResults):
    a = anchor_obj
    ares = anchor_results
    sec.append(NoEscape(r'\begin{samepage}'))
    subheader(sec, 'Anchor Minimum Spacing and Edge Distances')
    with sec.create(MiniPage(width=r"4in", pos='t')) as mini:
        if ares.spacing_requirements.slab_thickness_ok:
            mini.append(NoEscape(r'''The member thickness meets the minimum required thickness:'''))
            with mini.create(Math(inline=False)) as m:
                m.append(NoEscape(r't_{slab} \geq h_{min} \rightarrow '))
                m.append(NoEscape(
                    rf'{a.concrete_props.t_slab:.2f} \geq {a.anchor_props.hmin:.2f} \quad \text{{\textcolor{{Green}}{{ \textbf{{\textsf{{OK}} }} }} }}'))
        else:
            mini.append(NoEscape(r'''The member thickness is insufficient for the chosen anchor:'''))
            with mini.create(Math(inline=False)) as m:
                m.append(NoEscape(rf't_{{slab}} = {a.concrete_props.t_slab:.2f} < h_{{min}} \rightarrow '))
                m.append(NoEscape(
                    rf'{a.anchor_props.hmin:.2f} \quad \text{{ \textcolor{{red}}{{\textbf{{\textsf{{NG}} }} }} }}'))


        mini.append(NoEscape(rf'''The acceptance criteria for spacing and edge distance provided by ESR-{a.anchor_props.info.esr:.0f} 
        are listed below corresponding to the given concrete thickness. The plot to the right shows the given anchor spacing and edge distances 
        against these criteria \\'''))
        with mini.create(Alignat(numbering=False, escape=False)) as m:
            m.append(rf'c_{{min}} = {a.anchor_props.c1:.2f} \text{{ in}} \text{{ for }} s \geq {a.anchor_props.s1:.2f} \text{{ in,}} \quad')
            # m.append(rf'\text{{for }}s\geq {a.s1:.2f} \text{{ in}}\\')
            m.append(rf's_{{min}} = {a.anchor_props.s2:.2f} \text{{ in}} \text{{ for }}c\geq {a.anchor_props.c2:.2f} \text{{ in}}\\')
            # m.append(rf'\text{{for }}c\geq {a.c2:.2f} \text{{ in}}\\')
        if ares.spacing_requirements.edge_and_spacing_ok:
            mini.append(NoEscape(
                r'\textcolor{Green}{\textbf{Anchor meets spacing and edge distance requirements.}}'))


    sec.append(NoEscape(r'\hfill'))
    # fig, width = plots.anchor_spacing_criteria(a.c1, a.s1,
    #                                            a.c2, a.s2,
    #                                            a.c_min, a.s_min)
    # file = plots.plt_save()
    fig_name = 'spacing_crit'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file)

    if not ares.spacing_requirements.ok:
        sec.append(NoEscape(r'\bigskip'))
        sec.append(NewLine())
        sec.append(NoEscape(
            r'\textcolor{red}{\textbf{Anchor does not meet member thickness and/or spacing requirements!}}'))
    sec.append(NoEscape(r'\end{samepage}'))

def anchor_tension(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict,
                          anchor_obj: conc.ConcreteAnchors, results: conc.ConcreteAnchorResults):
    a = anchor_obj
    if a.anchor_props.Nsa:
        sec.append(NoEscape(rf'Tensile strength reported by manufacturer\
                            (see ESR {a.anchor_props.info.esr:.0f}): $N_{{sa}} = {a.anchor_props.Nsa:.0f}$ lbs'))
        # with sec.create(Flalign()) as fl:
        #     fl.append(NoEscape(
        #         rf'&N_{{sa}} = {a.Nsa} \text{{ lbs}} && \text{{Manufacturer-Provided Value. See ESR: {a.esr:.0f}}}'))
    else:
        # Todo: provide report option for non-tabulated Nsa
        raise Exception('Need to provide report functionality for non-tabulated Nsa')


def tension_breakout(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict,
                          anchor_obj: conc.ConcreteAnchors, results: conc.ConcreteAnchorResults):
    a = anchor_obj
    tg = results.tension_groups[results.governing_tension_group]
    calc = results.tension_breakout_calcs[results.governing_tension_group]
    idx_theta = results.governing_theta_idx

    with sec.create(MiniPage(width=NoEscape('2.5in'),pos='t')) as leftcol:
        # Input Paramters Table
        leftcol.append(NoEscape(r'\begin{footnotesize}'))
        with leftcol.create(Tabularx('Xcc', pos='t')) as table:
            table.add_hline()
            header = [NoEscape(r'\rowcolor{lightgray} Input Parameters'), '', '']
            table.add_row(header)
            table.add_hline()
            table.add_row(['Effective embedment (in)', NoEscape('$h_{ef}$'), NoEscape(f'{calc.hef}')])
            table.add_hline()
            table.add_row(['Edge Distances (in)', NoEscape('$c_{a,x+}$'), NoEscape(f'{tg.cax_pos:.2f}')])
            table.add_row(['', NoEscape('$c_{a,x-}$'), NoEscape(f'{tg.cax_neg:.2f}')])
            table.add_row(['', NoEscape('$c_{a,y+}$'), NoEscape(f'{tg.cay_pos:.2f}')])
            table.add_row(['', NoEscape('$c_{a,y-}$'), NoEscape(f'{tg.cay_neg:.2f}')])
            table.add_hline()
            table.add_row(['Critical Edge Dist. (in)', NoEscape('$c_{ac}$'), NoEscape(f'{a.anchor_props.cac:.2f}')])
            table.add_hline()
            table.add_row(['Effectivness Factor', NoEscape('$k_c$'), NoEscape(f'{a.anchor_props.kc:.1f}')])
            table.add_hline()
        leftcol.append(NoEscape(r'\end{footnotesize}'))
        leftcol.append(NewLine())

        # Breakout Figure
        fig_name = 'diagram'
        width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
        make_figure(leftcol, width, file)

    sec.append(NoEscape(r'\hfill'))
    with sec.create(MiniPage(width=NoEscape('3.75in'),pos='t')) as rightcol:
        rightcol.append(NoEscape(r'{\footnotesize \textbf{ \textit{Calculations}}}'))
        # rightcol.append(NewLine())
        rightcol.append(NoEscape(r'{\everydisplay{\tiny}'))  # open group and override display font size

        with rightcol.create(Flalign()) as fl:
            # ANco
            fl.append(NoEscape(r'&A_{Nco} &&= 9h_{ef}^2'))
            fl.append(NoEscape(
                rf'=9({calc.hef:.1f})^2 &&={calc.Anco:.1f} \text{{ in}}^2 &&\text{{ACI318-19 (17.6.2.1.4)}}\\'))

            # ANc
            fl.append(NoEscape(rf'''&A_{{Nc}} &&=({calc.bxN:.1f})\times ({calc.byN:.1f}) 
            && = {calc.Anc:.1f} \text{{ in}}^2 && \text{{ACI318-19 17.6.2.1.2}}\\ '''))

            # Psi,ecN (Eccentricity Factor)
            fl.append(rf'''&e_{{Nx}} &&= \frac{{\sum{{ \left(x_i-\bar{{x}}\right) T_i }}}}{{ \sum{{T_i}} }}
                &&={calc.ex[idx_theta]:.2f} \text{{ in}}\\''')
            fl.append(rf'''&\psi_{{ec,Nx}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{Nx}} }}{{1.5h_{{ef}} }}\right)}}
            && ={calc.psi_ecNx[idx_theta]:.2f}\text{{ in}} && \text{{ACI318-19 (17.6.2.3.1)}}\\''')

            fl.append(rf'''&e_{{Ny}} &&= \frac{{\sum{{ \left(y_i-\bar{{y}}\right) T_i }}}}{{ \sum{{T_i}} }}
                &&={calc.ey[idx_theta]:.2f} \text{{ in}}\\''')
            fl.append(rf'''&\psi_{{ec,Ny}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{Ny}} }}{{1.5h_{{ef}} }}\right)}}
            && ={calc.psi_ecNy[idx_theta]:.2f}\text{{ in}} && \text{{ACI318-19 (17.6.2.3.1)}}\\''')

            fl.append(rf'''&\psi_{{ec,N}} && =\psi_{{ec,Nx}} \times \psi_{{ec,Ny}} &&={calc.psi_ecN[idx_theta]:.2f}\\''')

            # Psi,ed (Edge Factor)
            fl.append(rf'''&\psi_{{ed,N}} && = \min{{\left(1.0,\quad 0.7 + 0.3\frac {{ c_{{a,min}}}}{{1.5h_{{ef}} }} \right)}}
                &&={calc.psi_edN:.2f} &&\text{{ACI318-19 (17.6.2.4.1)}}\\''')

            # Psi,cN (Breakout cracking factor)
            if calc.psi_cN:
                fl.append(rf'''&\psi_{{c,N}} &&  
                &&={calc.psi_cN:.2f} && \text{{ESR: {a.anchor_props.info.esr:.0f}}}\\''')
            else:
                # Todo: provide report option for non-tabulated Breakout cracking factor
                raise Exception('Need to provide report functionality for non-tabulated psi_cN')

            # Psi,cpN (Breakout splitting factor)
            fl.append(
                r'&\psi_{cp,N} && = \min{\left(1.0, \max{\left(\frac{c_{a,min}}{c_{ac}},\frac{1.5h_{ef}}{c_{ac}}\right)}\right)}')
            fl.append(rf' &&={calc.psi_cpN:.2f} &&\text{{ACI318-19 (17.6.2.6.1)}}\\')

            # Nb
            if all([a.geo_props.n_anchor == 1,
                    a.anchor_props.info.anchor_type in ['Headed Stud', 'Headed Bolt'],
                    11 <= calc.hef <= 25]):
                fl.append(rf'''&N_b && =16\lambda_a\sqrt{{f\prime _c}} h_{{ef}}^{{5/3}}
                && =16({a.concrete_props.lw_factor_a})\sqrt{{ ({a.concrete_props.fc}) }}({calc.hef})^{{5/3}}
                ={a.Nb:.2f} \text{{ lbs}} && \text{{ACI318-19 (17.6.2.2.3)}}\\''')
            else:
                fl.append(rf'''&N_b && =k_c\lambda_a\sqrt{{f_c^\prime}} h_{{ef}}^{{1.5}}
                            =({a.anchor_props.kc})({a.concrete_props.lw_factor_a})\sqrt{{ ({a.concrete_props.fc}) }}({calc.hef})^{{1.5}}
                            &&={calc.Nb:.2f} \text{{ lbs}} && \text{{ACI318-19 (17.6.2.2.1)}}\\''')

            # Ncb
            fl.append(r'&N_{cb} &&= \frac{A_{Nc}}{A_{Nco}}\psi_{ec,N}\psi_{ed,N}\psi_{c,N}\psi_{cp,N}N_b')
            fl.append(rf'&&={calc.Ncb[idx_theta]:.2f} \text{{ lbs}} && \text{{ACI318-19 (17.6.2.1)}}')
        rightcol.append(NoEscape(r'}')) # Close local font size group

def tension_pullout(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict,
                          anchor_obj: conc.ConcreteAnchors, results: conc.ConcreteAnchorResults):
    a = anchor_obj
    tg = results.tension_groups[results.governing_tension_group]
    calc = results.anchor_pullout_calcs[results.governing_tension_group]

    if calc.Np:
        sec.append(NoEscape(rf'Pull-out strength reported by manufacturer\
                            (see ESR {a.anchor_props.info.esr:.0f}): $N_{{p}} = {calc.Np:.0f}$ lbs'))
        # with sec.create(Flalign()) as fl:
        #     fl.append(NoEscape(
        #         rf'&N_{{sa}} = {a.Nsa} \text{{ lbs}} && \text{{Manufacturer-Provided Value. See ESR: {a.esr:.0f}}}'))
    else:
        # Todo: provide report option for non-tabulated Np
        raise Exception('Need to provide report functionality for non-tabulated Np')


def side_face_blowout(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict,
                          anchor_obj: conc.ConcreteAnchors, results: conc.ConcreteAnchorResults):
    a = anchor_obj
    # todo: add side face blowout section (for headed studs)


def bond_strength(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict,
                          anchor_obj: conc.ConcreteAnchors, results: conc.ConcreteAnchorResults):
    a = anchor_obj
    # todo: add bond strength section (for epoxy base_anchors)


def anchor_shear(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict,
                          anchor_obj: conc.ConcreteAnchors, results: conc.ConcreteAnchorResults):
    a = anchor_obj
    if a.anchor_props.Vsa:
        sec.append(NoEscape(rf'Anchor shear strength reported by manufacturer\
                            (see ESR {a.anchor_props.info.esr:.0f}): $V_{{sa}} = {a.anchor_props.Vsa:.0f}$ lbs'))
    else:
        # Todo: provide report option for non-tabulated Vsa
        raise Exception('Need to provide report functionality for non-tabulated Vsa')


def shear_breakout(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict,
                          anchor_obj: conc.ConcreteAnchors, results: conc.ConcreteAnchorResults):
    a = anchor_obj
    sg = results.shear_groups[results.governing_shear_group]
    calc = results.shear_breakout_calcs[results.governing_shear_group]
    idx_theta = results.governing_theta_idx

    with sec.create(MiniPage(width=NoEscape('2.5in'), pos='t')) as leftcol:
        # Input Parameter Table
        leftcol.append(NoEscape(r'\begin{footnotesize}'))
        with leftcol.create(Tabularx('Xcc', pos='t')) as table:
            table.add_hline()
            table.add_row([NoEscape(r'\rowcolor{lightgray} Input Parameters'), '', ''])
            table.add_hline()
            table.add_row(['Effective embedment (in)', NoEscape('$h_{a,eff}$'), NoEscape(f'{calc.ha_eff}')])
            table.add_hline()
            table.add_row(['Edge Distances (in)',
                           NoEscape('$c_{a1}$'), NoEscape(f'{sg.ca1:2f}')])
            table.add_row(['', NoEscape('$c_{a2+}$'), NoEscape(f'{sg.ca2p:.2f}')])
            table.add_row(['', NoEscape('$c_{a2-}$'), NoEscape(f'{sg.ca2n:.2f}')])
            table.add_hline()
            table.add_row(['Load Bearing Length (in)', NoEscape('$l_{e}$'), NoEscape(f'{a.anchor_props.le}')])
            table.add_hline()
            table.add_row(['Anchor Diameter (in)', NoEscape('$d_a$'), NoEscape(f'{a.anchor_props.da}')])
            table.add_hline()
        leftcol.append(NoEscape(r'\end{footnotesize}'))
        leftcol.append(NewLine())

        #Breakout Figure
        fig_name = 'diagram'
        width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
        make_figure(leftcol, width, file)


    sec.append(NoEscape(r'\hfill'))
    with sec.create(MiniPage(width=NoEscape('3.75in'),pos='t')) as rightcol:
        rightcol.append(NoEscape(r'{\footnotesize \textbf{ \textit{Calculations}}}'))
        rightcol.append(NewLine())
        rightcol.append(NoEscape(r'{\everydisplay{\tiny}'))  # open group and override display font size

        with rightcol.create(Flalign()) as fl:
            # AVco
            fl.append(r'&A_{Vco} &&= 4.5c_{a1}^2')
            fl.append(rf'''=4.5({sg.ca1:.2f})^2 &&={calc.Avco:.2f} \text{{ in}}^2 
                             &&\text{{ACI318-19 (17.7.2.1.3)}}\\''')

            # AVc
            fl.append(rf'''&A_{{Vc}} &&=({calc.b:.2f})\times ({calc.ha_eff:.2f}) 
                            && = {calc.Avc:.2f} \text{{ in}}^2 && \text{{ACI318-19 17.7.2.1.1}}\\ ''')

            # Psi,ecN (Eccentricity Factor)
            if sg.direction in [conc.ShearDirLabel.YP, conc.ShearDirLabel.YN]:
                fl.append(rf'''&e_{{V}} &&= \frac{{\sum{{ \left(x_i-\bar{{x}}\right) V_{{yi}} }}}}{{ \sum{{V_{{yi}}}} }}
                                    &&={calc.ecc[idx_theta]:.2f} \text{{ in}}\\''')
                fl.append(rf'''&\psi_{{ec,V}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{V}} }}{{1.5c_{{a1}} }}\right)}}
                                && ={calc.psi_ecV[idx_theta]:.2f}\text{{ in}} && \text{{ACI318-19 (17.7.2.3.1)}}\\''')
            else:
                fl.append(rf'''&e_{{V}} &&= \frac{{\sum{{ \left(y_i-\bar{{y}}\right) V_{{yi}} }}}}{{ \sum{{V_{{yi}}}} }}
                                    &&={calc.ecc[idx_theta]:.2f} \text{{ in}}\\''')
                fl.append(rf'''&\psi_{{ec,V}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{V}} }}{{1.5c_{{a1}} }}\right)}}
                                && ={calc.psi_ecV[idx_theta]:.2f}\text{{ in}} && \text{{ACI318-19 (17.7.2.3.1)}}\\''')

            # Psi,ed (Edge Factor)
            fl.append(rf'''&\psi_{{ed,V}} && = \min{{\left(1.0,\quad 0.7 + 0.3\frac {{ c_{{a2,min}}}}{{1.5h_{{ef}} }} \right)}}
                                &&={calc.psi_edV:.2f} &&\text{{ACI318-19 (17.7.2.4.1)}}\\''')

            # Psi,cV (Breakout cracking factor)
            fl.append(rf'''&\psi_{{c,V}} &&
                            &&={calc.psi_cV:.2f} &&\text{{ACI318-19 17.7.2.5.1}}\\''')

            # Psi,hV (Breakout thickenss factor)
            fl.append(
                r'&\psi_{h,V} && = \max{\left(1.0,\quad \sqrt{\frac{1.5c_{a1}}{h_{a}}}\right)}')
            fl.append(rf' &&={calc.psi_hV:.2f} &&\text{{ACI318-19 (17.7.2.6.1)}}\\')

            # Vb
            fl.append(r'''&V_b &&= \min{ \left( 7 \left(\frac{l_e}{d_a}\right)^{0.2}\sqrt{d_a},\quad 9\right)}
                            \lambda_a \sqrt{f_c'}\left(c_{a1}\right)^{1.5}''')
            fl.append(
                rf'&& = {calc.Vb:.2f} \text{{ lbs}} && \text{{ACI318-19 (17.7.2.2.1)}}\\')

            # # Vcb
            fl.append(r'&V_{cb} &&= \frac{A_{Vc}}{A_{Vco}}\psi_{ec,V}\psi_{ed,V}\psi_{c,V}\psi_{h,V}V_b')
            fl.append(
                rf'&&={calc.Vcb[idx_theta]:,.2f} \text{{ lbs}} && \text{{ACI318-19 (17.7.2.1)}}')
        rightcol.append(NoEscape(r'}')) # Close local font size group
    sec.append(NewLine())

    sec.append(NoEscape(r'{\footnotesize'))
    sec.append(NoEscape('''Note: ACI318 specifies that unless base anchors are welded to a common base plate,
        anchor breakout conditions must be considered in which a subset of anchors (with their respective proportion 
        of the toal load) forms a breakout cone at a lower capacity than the breakout cone of the entire group.
         In these calculations, all possible shear breakout groups were evaluated, with the governing case presented here.'''))
    sec.append(NoEscape(r'}'))


def shear_pryout(model_record: ModelRecord, sec, sec_title, sub_title, plots_dict,
                          anchor_obj: conc.ConcreteAnchors, results: conc.ConcreteAnchorResults):
    a = anchor_obj
    tg = results.tension_groups[results.governing_tension_group]
    calc = results.shear_pryout_calcs[results.governing_tension_group]
    idx_theta = results.governing_theta_idx

    with sec.create(Flalign()) as fl:
        fl.append(r'& k_{cp} &&= \begin{cases} '
                  r'1.0 & \text{for } h_{ef} < 2.5 \text{ in}\\'
                  r'2.0 & \text{for } h_{ef} \geq 2.5 \text{ in} '
                  r'\end{cases}')
        fl.append(rf'&&={calc.kcp}\\')
        fl.append(r'& V_{cp} && = k_{cp}N_{cp}')
        fl.append(rf'&&={calc.Vcp[idx_theta]:,.2f}\text{{ lbs}} &&\text{{ACI318-19 (17.7.3.1)}}')


def cmu_summary_full(item, sec, sec_title, sub_title, plots_dict):
    sec.append('See attached calculations for anchorage checks to CMU.')


def model_instability(item, sec, sec_title, sub_title, plots_dict):
    sec.append(NoEscape(
        r'\textcolor{red}{\textbf{\textsf{AN INSTABILITY EXISTS IN THE STRUCTURAL MODEL. REVIEW MODEL GEOMETRY, MATERIAL DEFINITIONS, AND FORCE RELEASES FOR MODEL COMPLETNESS.}}}'))


EquipmentReportSections.initialize_plots_list()
