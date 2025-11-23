# Define the refactored code based on the given objectives

from pylatex import (Package, LineBreak, NewLine, MiniPage, Tabular, LongTable,
                     Document, Command, FlushRight, FlushLeft, LargeText,
                     MediumText,NewPage, Section, Subsection, Subsubsection,
                     Tabularx, Math, MultiColumn, Alignat, Enumerate, MultiRow)

from pylatex.utils import bold, NoEscape
from pylatex.base_classes import Environment

from anchor_pro.concrete_anchors import ConcreteCMU, ConcreteAnchors, CMUAnchors
from anchor_pro.equipment import SMSAnchors
import anchor_pro.plots as plots
import anchor_pro.config

import numpy as np
import pandas as pd
import os

# import anchor_pro.calculator
# from io import BytesIO
# import tempfile
# import multiprocessing as mp
# import queue

import re

from anchor_pro.elements.wood_fasteners import WoodFastener

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

    def __init__(self, project_info, item_dict, governing_items, group_dict, pool=None):
        super().__init__(project_info, pool=pool)
        self.item_dict = item_dict
        self.governing_items = governing_items
        self.group_dict = group_dict
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
        item_sections_dict = {}
        for equipment_id, (group, governing_idx) in self.governing_items.items():
            item = self.item_dict[equipment_id]
            if group:
                sec_name = group
                group_items = [self.item_dict[eq_id] for eq_id in self.group_dict[group]]
            else:
                sec_name = equipment_id
                group_items = None
                sec_name = sec_name.rstrip()
                if item.equipment_type:
                    sec_name += f' [{item.equipment_type}]'

            item_sections_dict[sec_name] = EquipmentReportSections(item,
                                                                   group_name=group,
                                                                   group_items=group_items,
                                                                   group_idx=governing_idx)
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
                        fig, width = plot_func(report_section.item, *sec_pars['args'], filename=filename)
                        file = plots.vtk_save(fig, filename=filename)
                    else:
                        fig, width = plot_func(report_section.item, *sec_pars['args'])
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
            item = report_section.item
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
                        func(item, sec, sec_title, sub_title, self.plots_dict, *args, **kwargs)


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
            equilibrium_solution.__name__: {},
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

    def __init__(self, item, group_name=None, group_items=None, group_idx=0):
        super().__init__()
        self.item = item
        self.group_name = group_name
        self.group_items = group_items
        self.group_idx = group_idx
        self.governing_backing = None
        self.wall_anchors = None

        self.sections_dict = self.initialize_sections_dictionary()
        # self.set_section_hierarchy()
        self.set_section_inclusions(item.code_edition)
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
            ('ASD Factored Loads', asd_loads, 2),
            ('Equilibrium Solution', equilibrium_solution, 2),
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

    def set_section_inclusions(self, code):
        item = self.item
        sd = self.sections_dict

        # Front and End Matter
        sd['Introduction']['include'] = bool(item.frontmatter_file)
        sd['Addendum']['include'] = bool(item.endmatter_file)

        # Group Summary
        sd['Group Summary']['include'] = self.group_items is not None

        # Basic Report Elements
        sd['Unit Summary']['include'] = True

        if self.item.model_unstable:
            sd['WARNING: Model Instability']['include'] = True
            return

        if not self.item.omit_analysis:
            # Load Sections
            sd['Equipment Loads']['include'] = True
            sd['Seismic Load $F_p$ (ASCE 7-16)']['include'] = code == 'ASCE 7-16'
            sd['Maximum Assumed Seismic Load $F_p$']['include'] = code == 'ASCE 7-22 OPM'
            sd['Seismic Load $F_p$ (CBC 1998)']['include'] = code == 'CBC 1998, 16B'

            sd['LRFD Factored Loads']['include'] = code in ['ASCE 7-16', 'ASCE 7-22 OPM']
            sd['LRFD Factored Loads (CBC 1998)']['include'] = code == 'CBC 1998, 16B'
            sd['ASD Factored Loads']['include'] = False  # todo: currently never used

            sd['Equilibrium Solution']['include'] = False  # todo: has been made obsolete. Language included in LRFD section

            # Base Anchor Demands
            sd['Base Anchor Demands']['include'] = item.installation_type in ['Base Anchored', 'Wall Brackets'] \
                                                   and item.base_anchors is not None

        # Base Concrete Anchor Checks
        has_base_concrete_anchors = bool(item.base_anchors) \
                                    and isinstance(item.base_anchors, ConcreteAnchors)

        if has_base_concrete_anchors:
            sd['Base Concrete Anchor Checks']['include'] = True
            if not all(item.base_anchors.spacing_requirements.values()):
                sd['Base Anchor Spacing Limits']['include'] = True
            elif not item.omit_analysis:
                sd['Base Concrete Anchor Summary']['include'] = True
                limit_state_to_section_name_map = {
                    'Steel Tensile Strength': 'Base Anchor in Tension [ACI318-19, 17.6.1]',
                    'Concrete Tension Breakout': 'Base Anchor Tension Breakout [ACI318-19, 17.6.2]',
                    'Anchor Pullout': 'Base Anchor Pullout [ACI318-19, 17.6.3]',
                    'Side Face Blowout': 'Base Anchor Side-Face Blowout [ACI318-19, 17.6.4]',
                    'Bond Strength': 'Base Anchor Bond Strength [ACI318-19, 17.6.5]',
                    'Steel Shear Strength': 'Base Anchor in Shear [ACI318-19, 17.7.1]',
                    'Shear Breakout (X+)': 'Base Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Edge Breakout (X+)': 'Base Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Breakout (X-)': 'Base Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Edge Breakout (X-)': 'Base Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Breakout (Y+)': 'Base Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Edge Breakout (Y+)': 'Base Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Breakout (Y-)': 'Base Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Edge Breakout (Y-)': 'Base Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Pryout': 'Base Anchor Shear Pryout [ACI318-19, 17.7.3]'}

                short_name_to_long_name = {v: k for k, v in
                                           item.base_anchors.shear_breakout_long_name_to_short_name_map.items()}
                for limit_name, row in item.base_anchors.results.iterrows():
                    # Logic to exclude any non-governing shear breakout cases to avoid redundancy in the report
                    case1 = (row['Mode'] == 'Tension')
                    case2 = (row['Mode'] == 'Shear') and ('Breakout' not in limit_name)
                    case3 = ((row['Mode'] == 'Shear') and
                             (item.base_anchors.governing_shear_breakout_case is not None) and
                             (limit_name == short_name_to_long_name[item.base_anchors.governing_shear_breakout_case]))
                    if any([case1, case2, case3]):
                        sd[limit_state_to_section_name_map[limit_name]]['include'] = True

        # Base Connection Demands
        has_base_connections = len([plate.connection_forces for plate in item.floor_plates if
                                    not all([plate.x0 == 0, plate.y0 == 0, plate.z0 == 0])]) > 0
        if has_base_connections and not self.item.omit_analysis:
            sd['Base Plate Connections']['include'] = True

            # Base SMS Attachment
            sd['Base Connection SMS Demand']['include'] = sd['Base Connection SMS Checks']['include'] = \
                any([isinstance(plate.connection.anchors_obj, SMSAnchors)
                     for plate in item.floor_plates if plate.connection])

        # Base Straps
        sd['Base Straps']['include'] = len(item.base_straps) > 0 and not self.item.omit_analysis

        # Wall Brackets and Wall Anchors
        has_brackets = item.installation_type in ['Wall Brackets', 'Wall Mounted']
        if has_brackets and not self.item.omit_analysis:
            sd['Wall Fastener Demand']['include'] = True
            sd['Wall Brackets']['include'] = \
                sd['Wall Bracket Demand']['include'] = \
                sd['Wall Bracket Checks']['include'] = not self.item.omit_bracket_output
            # sd['Wall Anchors']['include'] = \
            self.governing_backing = max(item.wall_backing, key=lambda obj: obj.anchor_forces[:, 0].max())

        # Wall Concrete Anchors
        wall_anchors_list = [anchors for wall, anchors in item.wall_anchors.items() if anchors is not None]
        has_wall_concrete_anchors = any(
            [isinstance(anchors, ConcreteAnchors) for anchors in wall_anchors_list])
        if has_wall_concrete_anchors:
            if not self.item.omit_analysis:
                self.wall_anchors = self.item.wall_anchors[self.governing_backing.supporting_wall]
            else:
                self.wall_anchors = next((wall_anchors for wall_anchors in wall_anchors_list if
                                          all(wall_anchors.spacing_requirements.values())),
                                         wall_anchors_list[-1])
            sd['Wall Concrete Anchor Checks']['include'] = True
            if not all(self.wall_anchors.spacing_requirements.values()):
                sd['Wall Anchor Spacing Limits']['include'] = True
            elif not item.omit_analysis:
                sd['Wall Concrete Anchor Summary']['include'] = True
                limit_state_to_section_name_map = {
                    'Steel Tensile Strength': 'Wall Anchor in Tension [ACI318-19, 17.6.1]',
                    'Concrete Tension Breakout': 'Wall Anchor Tension Breakout [ACI318-19, 17.6.2]',
                    'Anchor Pullout': 'Wall Anchor Pullout [ACI318-19, 17.6.3]',
                    'Side Face Blowout': 'Wall Anchor Side-Face Blowout [ACI318-19, 17.6.4]',
                    'Bond Strength': 'Wall Anchor Bond Strength [ACI318-19, 17.6.5]',
                    'Steel Shear Strength': 'Wall Anchor in Shear [ACI318-19, 17.7.1]',
                    'Shear Breakout (X+)': 'Wall Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Edge Breakout (X+)': 'Wall Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Breakout (X-)': 'Wall Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Edge Breakout (X-)': 'Wall Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Breakout (Y+)': 'Wall Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Edge Breakout (Y+)': 'Wall Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Breakout (Y-)': 'Wall Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Edge Breakout (Y-)': 'Wall Anchor Shear Breakout [ACI318-19, 17.7.2]',
                    'Shear Pryout': 'Wall Anchor Shear Pryout [ACI318-19, 17.7.3]'}

                short_name_to_long_name = {v: k for k, v in
                                           self.wall_anchors.shear_breakout_long_name_to_short_name_map.items()}
                for limit_name, row in self.wall_anchors.results.iterrows():
                    # Logic to exclude any non-governing shear breakout cases to avoid redundancy in the report
                    case1 = (row['Mode'] == 'Tension')
                    case2 = (row['Mode'] == 'Shear') and ('Breakout' not in limit_name)
                    case3 = ((row['Mode'] == 'Shear') and
                             (self.wall_anchors.governing_shear_breakout_case is not None) and
                             (limit_name == short_name_to_long_name[
                                 self.wall_anchors.governing_shear_breakout_case]))
                    if any([case1, case2, case3]):
                        sd[limit_state_to_section_name_map[limit_name]]['include'] = True

        # Wall CMU Anchors
        has_wall_cmu_anchors = any(
            [isinstance(anchors, CMUAnchors) for wall, anchors in item.wall_anchors.items()])
        if has_wall_cmu_anchors and not self.item.omit_analysis:
            sd['Wall CMU Anchor Checks']['include'] = True

        # Wall SMS Anchors
        has_wall_sms_anchors = any([isinstance(b.anchors_obj, SMSAnchors)
                                    for b in item.wall_backing])
        if has_wall_sms_anchors and not self.item.omit_analysis:
            self.wall_anchors = self.governing_backing.anchors_obj
            sd['Wall SMS Checks']['include'] = True

        # Wall Wood Fasteners
        has_wall_wood_fasteners = any([isinstance(b.anchors_obj, WoodFastener)
                                    for b in item.wall_backing])
        if has_wall_wood_fasteners and not self.item.omit_analysis:
            self.wall_anchors = self.governing_backing.anchors_obj
            sd['Wall Fastener Checks']['include'] = True


        # Bracket Connection Demands
        has_bracket_connections = any([bracket.connection is not None for bracket in item.wall_brackets ])
        if has_bracket_connections and not self.item.omit_analysis:
            sd['Bracket Connections']['include'] = True

            # Bracket SMS Attachment
            sd['Bracket Connection SMS Demand']['include'] = sd['Bracket Connection SMS Checks']['include'] = \
                any([isinstance(bracket.connection.anchors_obj, SMSAnchors)
                     for bracket in item.wall_brackets if bracket.connection is not None])

    def set_section_titles(self):
        if self.sections_dict['Wall Brackets']['include']:
            self.sections_dict['Wall Brackets']['alt_title'] = f'{self.item.wall_brackets[0].bracket_id} (Wall Bracket)'
            self.sections_dict['Wall Bracket Demand']['alt_title'] = f'{self.item.wall_brackets[0].bracket_id} Demand'
            self.sections_dict['Wall Bracket Checks']['alt_title'] = f'{self.item.wall_brackets[0].bracket_id} Checks'

    def set_section_args(self):
        """ This provides any additional arguments used by the section functions beyond the standard
        item, sec, sec_title, sub_title, plots_dict"""

        item = self.item
        sd = self.sections_dict

        # Group Summary
        sd['Group Summary']['args'] = (self.group_name, self.group_items, self.group_idx)

        # Base SMS Connection
        if sd['Base Connection SMS Demand']['include']:
            sol = item.governing_solutions['base_anchor_tension']['sol']
            item.update_element_resultants(sol)
            plates_with_connections = [plate.connection for plate in item.floor_plates if plate.connection is not None]

            # Use enumerate to keep track of the index in the filtered list
            governing_cxn_idx, base_connection_obj = max(
                enumerate(plates_with_connections),
                key=lambda pair: pair[1].anchors_obj.max_dcr()
            )
            # base_connection_obj = max([plate.connection for plate in item.floor_plates if plate.connection is not None],
            #                           key=lambda x: x.anchors_obj.max_dcr())
            sd['Base Plate Connections']['args'] = (governing_cxn_idx,)
            sd['Base Connection SMS Demand']['args'] = (base_connection_obj, 'base')
            sd['Base Connection SMS Checks']['args'] = (base_connection_obj.anchors_obj,)

        # Bracket SMS Connection
        if sd['Bracket Connections']['include']:
            sol = item.governing_solutions['wall_bracket_tension']['sol']
            item.update_element_resultants(sol)
            bracket_connection_obj = max([bracket.connection for bracket in item.wall_brackets],
                                         key=lambda x: x.anchors_obj.max_dcr())
            sd['Bracket Connection SMS Demand']['args'] = (bracket_connection_obj, 'bracket')
            sd['Bracket Connection SMS Checks']['args'] = (bracket_connection_obj.anchors_obj,)

        # Base Concrete Anchors
        profile = 'Slab'  # todo [Fill over Deck] make profile dynamic.
        sd['Base Anchor Spacing Limits']['args'] = \
            sd['Base Concrete Anchor Summary']['args'] = (item.base_anchors, profile)
        conc_limit_names_list = ['Anchor in Tension [ACI318-19, 17.6.1]',
                                 'Anchor Tension Breakout [ACI318-19, 17.6.2]',
                                 'Anchor Pullout [ACI318-19, 17.6.3]',
                                 'Anchor Side-Face Blowout [ACI318-19, 17.6.4]',
                                 'Anchor Bond Strength [ACI318-19, 17.6.5]',
                                 'Anchor in Shear [ACI318-19, 17.7.1]',
                                 'Anchor Shear Breakout [ACI318-19, 17.7.2]',
                                 'Anchor Shear Pryout [ACI318-19, 17.7.3]']
        for limit_name in conc_limit_names_list:
            sd['Base ' + limit_name]['args'] = (item.base_anchors,)

        # Wall Anchor Demands
        sd['Wall Fastener Demand']['args'] = (self.governing_backing,)

        # Wall Concrete Anchors
        profile = 'Wall'
        sd['Wall Anchor Spacing Limits']['args'] = \
            sd['Wall Concrete Anchor Summary']['args'] = (self.wall_anchors, profile)
        for limit_name in conc_limit_names_list:
            sd['Wall ' + limit_name]['args'] = (self.wall_anchors,)

        # Wall SMS Anchors
        sd['Wall SMS Checks']['args'] = (self.wall_anchors,)

        # Wall Wood Fasteners
        sd['Wall Fastener Checks']['args'] = (self.wall_anchors,)


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


def description(item, sec, sec_title, sub_title, plots_dict):
    # with sec.create(Subsection("Description")) as ss:
    # sec.append("The base geometry and attachment points are indicated below.")
    # sec.append(NoEscape(r'\bigskip'))
    # sec.append(NewLine())

    with sec.create(Tabularx('lX', pos='t')) as table:

        table.add_hline()
        header = [NoEscape(r'\rowcolor{lightgray} Unit Parameters'), '']
        table.add_row(header)
        table.add_hline()
        table.add_row(['Equipment ID', item.equipment_id])
        table.add_hline()
        table.add_row([NoEscape('Max Operating Weight, $W_p$'), rf'{item.Wp:.2f} lbs'])
        table.add_row([NoEscape(r'Dimensions, $W\times D\times H$'),
                       NoEscape(rf'${item.Bx:.2f}$ in. $\times {item.By:.2f}$ in. $\times{item.H:.2f}$ in.')])
        if item.ex == item.ey == 0.0:
            table.add_row([NoEscape(r'Center of Gravity, $z_{CG}$'), NoEscape(rf'${item.zCG:.2f}$ in.')])
        else:
            table.add_row([NoEscape(r'Center of Gravity, $e_x$, $e_y$, $z_{CG}$'),
                           NoEscape(rf'${item.ex:.2f}$ in.,  ${item.ey:.2f}$ in., ${item.zCG:.2f}$ in.')])
        table.add_hline()
        if isinstance(item.base_anchors, ConcreteAnchors):
            table.add_row([NoEscape(r'\rowcolor{lightgray} Base Anchor and Substrate'), ''])
            table.add_hline()
            table.add_row(['Base Condition', 'Anchorage to Concrete'])
            table.add_row(['Anchor Type', NoEscape(rf'\textbf{{{item.base_anchors.anchor_id}}}')])
            table.add_row([NoEscape('$h_{ef}$'), rf'{item.base_anchors.hef_default}'])
            cracked_text = 'Cracked Concrete, ' if item.base_anchors.cracked_concrete else 'Uncracked Concrete, '
            fc_text = rf'$f^\prime_c = {item.base_anchors.fc:.0f}$ psi'
            table.add_row(['Base Material', NoEscape(cracked_text + fc_text)])
            table.add_row(['Base Thickness', f'{item.base_anchors.t_slab:.2f}'])
            if item.include_pull_test and not item.omit_analysis:
                item.update_element_resultants(item.governing_solutions['base_anchor_tension']['sol'])
                table.add_row(['Pull-test Load', f'{max([500, 3 * item.base_anchors.Tu_max]):.0f} lbs'])
            if item.base_straps:
                table.add_row(['Base Strap', item.base_strap])
            table.add_hline()
        if item.wall_brackets:
            table.add_row([NoEscape(r'\rowcolor{lightgray} Wall Fastener and Substrate'), ''])
            table.add_hline()
            table.add_row(['Wall Type', rf'{item.wall_type}'])
            if not item.omit_bracket_output:
                table.add_row(['Unit-to-Wall Hardware', item.wall_brackets[0].bracket_id])
            anchors = [anchors for wall, anchors in item.wall_anchors.items() if anchors is not None]
            anchors += [backing.anchors_obj for backing in item.wall_backing if backing.anchors_obj is not None]
            anchor_obj = max(anchors,key=lambda x: x.Tu_max)
            if isinstance(anchor_obj, ConcreteCMU):
                table.add_row(['Wall Fastener', NoEscape(rf'\textbf{{{item.wall_anchor_id}}}')])
                table.add_row([NoEscape('$h_{ef}$'), rf'{anchor_obj.hef_default}'])
                cracked_text = 'Cracked Concrete, ' if anchor_obj.cracked_concrete else 'Uncracked Concrete, '
                fc_text = rf'$f^\prime_c = {anchor_obj.fc:.0f}$ psi'
                table.add_row(['Wall Material', NoEscape(cracked_text + fc_text)])
                table.add_row(['Wall Thickness', f'{anchor_obj.t_slab:.2f}'])

            elif isinstance(anchor_obj, SMSAnchors):
                table.add_row(['Wall Fastener', NoEscape(rf'\textbf{{{item.wall_sms_id}}}')])
                table.add_row(['Wall Type', f'Steel Studs, {anchor_obj.gauge:.0f} GA, {anchor_obj.fy:.0f} ksi'])
            if item.include_pull_test and not item.omit_analysis:
                wall_anchors = []
                item.update_element_resultants(item.governing_solutions['wall_anchor_tension']['sol'])
                pull_test_text = f'{max([3 * anchor_obj.Tu_max, 500]):.0f} lbs' if anchor_obj.Tu_max else 'N/A'
                table.add_row(['Pull-test Load', pull_test_text])

            table.add_hline()

    sec.append(NoEscape(r'\bigskip'))
    sec.append(NewLine())
    with sec.create(MiniPage(width=r"6.5in", pos='t')) as mini:
        # Equipment Figure
        # subheader(mini,'Isometric')
        fig_name = '3d'
        width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
        make_figure(mini, 2, file,title='ISOMETRIC')

        mini.append(NoEscape(r'\hfill'))

        if item.base_anchors is not None:
            # Equipment Plan View
            fig_name = 'plan'
            width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
            make_figure(mini, 2, file,title='PLAN')
            mini.append(NoEscape(r'\hfill'))

        fig_name = 'elev_xz'
        width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
        make_figure(mini, 2, file, title='ELEVATION')

    # todo: add ASCE7 classification, once implimented in spreadsheet


def equipment_loads(item, sec, sec_title, sub_title, plots_dict):
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

def fp_cbc_1998(item, sec, sec_title, sub_title, plots_dict):
    with sec.create(Tabular('p{0.35\\textwidth} p{0.6\\textwidth}', pos='t')) as table:
        table.add_hline()
        header = [NoEscape(r'\rowcolor{lightgray} Unit Parameters'), '']
        table.add_row(header)
        table.add_hline()
        table.add_row(['Table 16B-O Category', item.code_pars['cp_category']])
        table.add_hline()
        table.add_row([NoEscape('$C_p$'), item.code_pars['Cp']])
        table.add_hline()
        table.add_row([NoEscape('Rigid/Flexible Factor, $R$'), item.code_pars['cp_amplification']])
        table.add_hline()
        table.add_row([NoEscape(r'At-or-below-grade Factor, $G$'),
                       f'{item.code_pars["grade_factor"]:.2f}'])
        table.add_hline()

    with sec.create(Flalign(numbering=False, escape=False)) as align:
        align.append(rf'''&C^\prime_{{p}}
            &&=C_p  R  G && \leq \begin{{cases}}
            2G & \text{{for }} R = 2\\
            3G & \text{{for }} R = 4
            \end{{cases}}  && = {item.code_pars['Cp_eff']}
            & \text{{CBC98 (\S1630B.2)}}\\''')
        align.append(rf'''&F_{{p}}
                    &&=ZI_pC^\prime_{{p}}W_p &&
                    &&={item.Fp:.2f} \text{{ lb}}
                    &\quad \text{{CBC98 (30B-1)}}\\''')
        align.append(
            rf'''&E_h &&= F_p && &&= {item.Eh:.2f} \text{{ lb}} &\text{{\hfill Horizontal Seismic Force}}\\''')
        align.append(
            rf'''&E_v &&= F_p/3 && && = {item.Ev:.2f} \text{{ lb}} &\text{{Vertical Seismic Force}}''')

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
    


def lrfd_loads_cbc_1998(item, sec, sec_title, sub_title, plots_dict):
    with sec.create(Flalign()) as align:

        align.append(
            rf'''&F_{{uh}} &&=0.75(1.7)(1.1E_h) &&=0.75(1.7)(1.1)({item.Eh:.2f}) 
            &&= {item.Fuh:.2f} \text{{ lb}} &\text{{CBC98 (9B-2)}}\\''')
        align.append(rf'''&F_{{uv,min}} &&=-0.9W_p+1.3(1.1E_v) &&=-0.9({item.Wp:.2f})+1.3(1.1)({item.Ev:.2f})
        &&={item.Fuv_min:.2f} \text{{ lb}} & \text{{CBC98 (9B-3)}}\\ ''')
        align.append(rf'''&F_{{uv,max}} &&=-0.75(1.4W_p+1.7(1.1E_v)) &&=-0.75(1.4{item.Wp:.2f})-1.7(1.1)({item.Ev:.2f})
                            &&={item.Fuv_max:.2f} \text{{ lb}} & \text{{CBC98 (9B-2)}} ''')

    _analysis_description_text(sec)

def _analysis_description_text(sec):
    sec.append(
        NoEscape(r'''The factored horizontal load, $F_{uh}$ is applied at angles, $0 \leq \theta_z \leq 360$. 
                An analytical model is used to determine distribution of applied loads to anchoring elements.'''))
def asd_loads(item, sec, sec_title, sub_title, plots_dict):
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

def equilibrium_solution(_, sec, sec_title, sub_title, plots_dict):
    # sec.append(
    #     'An analytical model is used to determine distribution of applied loads to anchoring elements. The equipment item is idealized as a rigid body with six degrees of freedom (3 translations and 3 rotations). The governing equations of equilibrium can be expressed as:')
    # sec.append(Math(data=["P", "=", "KU"]))
    # sec.append(NoEscape(
    #     'Where $P$ are the applied loads, $U$ are the DOF displacements, and $K$ is a stiffness matrix determined from the anchoring elements.'))
    # sec.append(
    #     'The DOF displacement solution is then used to calculate internal forces to anchors, brackets, or other supporting elements.')
    # sec.append(NoEscape(r'The factored horizontal load is applied at angles, $0 \leq \theta_z \leq 360$.'))

    sec.append(
        NoEscape(r'''The factored horizontal load, $F_{uh}$ is applied at angles, $0 \leq \theta_z \leq 360$. 
        An analytical model is used to determine distribution of applied loads to anchoring elements.'''))

# def no_prying(item,sec,sec_title, sub_title, plots_dict):
    # todo: continue here


# def prying_action(item,sec,sec_title, sub_title, plots_dict):



def base_anchor_demands(item, sec, sec_title, sub_title, plots_dict):
    sol = item.governing_solutions['base_anchor_tension']['sol']
    item.update_element_resultants(sol)

    # Anchor Forces Plot
    if item.base_anchors is not None:
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

        # Anchors Table
        if item.base_anchors is not None:
            title = 'Summary of Base Anchors(s)'
            header = [NoEscape('$x$'), NoEscape('$y$'), NoEscape('$K^{(+)}$'), NoEscape('$N$')]
            units = ['(in)', '(in)', '(lb/in)', '(lbs)']
            data = np.hstack((item.base_anchors.xy_anchors,
                              np.full((item.base_anchors.xy_anchors.shape[0], 1), item.base_anchors.K),
                              item.base_anchors.anchor_force_results[:, 0].reshape(-1, 1))) #todo add shear

            formats = ['{:.2f}', '{:.2f}', '{:.0f}', '{:.2f}']
            make_table(mini, title, header, units, data, col_formats=formats, width=r'\textwidth',
                       use_minipage=True, align='l',rows_to_highlight=item.base_anchors.group_idx)

        mini.append(NoEscape(r'\bigskip'))

        # Bearing Areas Table
        if item.floor_plates:  # todo [Testing]: Verify if this section works with anchors but no bearing boundaries
            mini.append(NoEscape(r'\bigskip'))
            mini.append(NewLine())
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
            for plate in item.floor_plates:
                cz_result = plate.cz_result
                for index in range(len(cz_result['areas'])):
                    data.append([
                        cz_result["centroids"][index][0],
                        cz_result["centroids"][index][1],
                        cz_result["areas"][index],
                        # cz_result["Ixx"][index],
                        # cz_result["Iyy"][index],
                        # cz_result["Ixy"][index],
                        cz_result["beta"][index],
                        -cz_result["fz"][index]])

            # formats = ['{:.2f}', '{:.2f}', '{:.2f}', '{:.2f}', '{:.2f}', '{:.2f}', '{:.0f}', '{:.0f}']
            formats = ['{:.2f}', '{:.2f}', '{:.2f}', '{:.0f}', '{:.0f}']
            make_table(mini, title, header, units, data, col_formats=formats, width=r'\textwidth', use_minipage=True,align='l')


def base_connection_demands(item, sec, sec_title, sub_title, plots_dict, governing_cxn_idx=None):
    # add_capacity_checks = {'sms_calcs': any([isinstance(plate.connection.anchors_obj, calculator.SMSAnchors)
    #                                          for plate in item.floor_plates if plate.connection]),
    #                        'weld_calcs': False,
    #                        'bolt_calcs': False}

    # Attachment Forces Table

    data = [np.concatenate([[plate.xc, plate.yc, plate.zc], plate.connection_forces]) for plate in item.floor_plates if
            not all([plate.xc == 0, plate.yc == 0, plate.zc == 0])]

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


def wall_brackets(_, sec, sec_title, sub_title, plots_dict):
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

def wall_bracket_demands(item, sec, sec_title, sub_title, plots_dict):
    sol = item.governing_solutions['wall_bracket_tension']['sol']
    item.update_element_resultants(sol)

    # Bracket Forces Plots
    title = f'{item.wall_brackets[0].bracket_id} Forces vs. Direction of Loading'
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
                 bracket.bracket_forces['fz']] for bracket in item.wall_brackets]
        highlight_idx = np.argmax([bracket.bracket_forces['fn'] for bracket in item.wall_brackets])
        formats = ['{:.2f}', '{:.2f}', '{:.2f}', '{:.0f}', '{:.0f}', '{:.0f}']
        alignment = 'cccccc'
        make_table(mini, title, header, units, data, col_formats=formats, alignment=alignment, width=r'3.5in',
                   rows_to_highlight=highlight_idx)


def wall_bracket_checks(item, sec, sec_title, sub_title, plots_dict):
    Tn = item.wall_brackets[0].bracket_capacity
    Rn_eq = item.wall_brackets[0].capacity_to_equipment
    Rn_backing = item.wall_brackets[0].capacity_to_backing
    check_bracket = isinstance(Tn, (int, float))
    check_to_eq = isinstance(Rn_eq, (int, float))
    check_to_backing = isinstance(Rn_backing, (int, float))

    if not any([check_bracket, check_to_eq, check_to_backing]):
        sec.append(
            "By inspection, bracket elements are determined not to be the governing component of the load path.")
        return
    Tu = max([b.bracket_forces['fn'] for b in item.wall_brackets])
    if item.wall_brackets[0].capacity_method == 'ASD':
        Ta = Tu*item.asd_lrfd_ratio
    
        sec.append(NoEscape(rf"The maximum bracket tension force for any angle of loading, converted to ASD-level demand is:\\ $T_a = T_u\left(\frac{{ F_{{h,ASD}} }}{{ F_{{h,LRFD}} }}\right) = {Ta:.2f}$\\"))
        # with sec.create(Math(inline=False)) as m:
        #     m.append(NoEscape(rf'T_u = {Tu:.2f}'))
        sec.append(
            "Capacities for bracket elements are taken from manufacturer data or pre-tabulated by the engineer.")
        sec.append(NewLine())
        if check_bracket:
            sec.append(NoEscape(rf'\textbf{{\textit{{{item.wall_brackets[0].bracket_id} Capacity Check}}}}'))
            with sec.create(Flalign()) as align:
                align.append(rf'&DCR &&=T_a/{{(T_n/\Omega()}} &&=\frac{{({Ta:.2f})}}{{({Tn})}} &&= {Tu / Tn:.2f}')
        if check_to_eq:
            sec.append(NoEscape(rf'\textbf{{\textit{{{item.wall_brackets[0].bracket_id} Connection to Equipment}}}}'))
            with sec.create(Flalign()) as align:
                align.append(rf'&DCR &&=T_a/{{(R_n/\Omega)}} &&=\frac{{({Tu:.2f})}}{{({Rn_eq})}} &&= {Tu / Rn_eq:.2f}')
        if check_to_backing:
            sec.append(NoEscape(rf'\textbf{{\textit{{{item.wall_brackets[0].bracket_id} Connection to Wall Backing}}}}'))
            with sec.create(Flalign()) as align:
                align.append(
                    rf'&DCR &&=T_a/{{(R_n/\Omega)}} &&=\frac{{({Tu:.2f})}}{{({Rn_backing})}} &&= {Tu / Rn_backing:.2f}')
    else:
        Tu = max([b.bracket_forces['fn'] for b in item.wall_brackets])
    
        sec.append(NoEscape(rf"The maximum bracket tension force for any angle of loading is: $T_u = {Tu:.2f}$\\"))
        # with sec.create(Math(inline=False)) as m:
        #     m.append(NoEscape(rf'T_u = {Tu:.2f}'))
        sec.append(
            "Capacities for bracket elements are taken from manufacturer data or pre-tabulated by the engineer.")
        sec.append(NewLine())
        if check_bracket:
            sec.append(NoEscape(rf'\textbf{{\textit{{{item.wall_brackets[0].bracket_id} Capacity Check}}}}'))
            with sec.create(Flalign()) as align:
                align.append(rf'&DCR &&=T_u/{{\phi T_n}} &&=\frac{{({Tu:.2f})}}{{({Tn})}} &&= {Tu / Tn:.2f}')
        if check_to_eq:
            sec.append(NoEscape(rf'\textbf{{\textit{{{item.wall_brackets[0].bracket_id} Connection to Equipment}}}}'))
            with sec.create(Flalign()) as align:
                align.append(rf'&DCR &&=T_u/{{\phi R_n}} &&=\frac{{({Tu:.2f})}}{{({Rn_eq})}} &&= {Tu / Rn_eq:.2f}')
        if check_to_backing:
            sec.append(NoEscape(rf'\textbf{{\textit{{{item.wall_brackets[0].bracket_id} Connection to Wall Backing}}}}'))
            with sec.create(Flalign()) as align:
                align.append(
                    rf'&DCR &&=T_u/{{\phi R_n}} &&=\frac{{({Tu:.2f})}}{{({Rn_backing})}} &&= {Tu / Rn_backing:.2f}')


def wall_anchor_demands(item, sec, sec_title, sub_title, plots_dict, governing_backing):
    sol = item.governing_solutions['wall_anchor_tension']['sol']
    item.update_element_resultants(sol)
    # governing_backing = max(item.wall_backing, key=lambda obj: obj.anchor_forces[:, 0].max())

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
    data = governing_backing.anchor_forces
    make_table(sec, "Governing Wall Fastener Forces", header, units, data, alignment='ccc', col_formats=formats,
               width='3in')


def bracket_connection_demands(item, sec, sec_title, sub_title, plots_dict):
    # Attachment Forces Table
    data = [bracket.connection_forces for bracket in item.wall_brackets if bracket.connection_forces]

    sol = item.governing_solutions['wall_bracket_tension']['sol']
    item.update_element_resultants(sol)

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
    make_table(sec, "Bracket Attachment Forces", header, units, data, col_formats=formats)
    sec.append(NewLine())


def sms_connection_demands(_, sec, sec_title, sub_title, plots_dict, connection_obj, connection_type):
    """section taking nodal demands and applying them to sms group"""
    type_to_text = {'base': 'Base plates',
                    'bracket': 'Bracket'}

    # sec.append(NoEscape(r'\bigskip'))
    # sec.append(NewLine())
    sec.append(NoEscape(
        f'{type_to_text[connection_type]} are attached to the equipment using sheet-metal-screws. '
        'Centroid forces (normal, shear and moments) on the connection screws group are computed '
        r'from nodal forces at the connection of the floor plate element to equipment model.\\'))

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
    header = [NoEscape('Normal, $N$'), NoEscape('Shear, $V$')]
    units = ['(lbs)', '(lbs)']
    formats = ['{:.0f}', '{:.0f}']
    data = np.column_stack((connection_obj.anchors_obj.anchor_forces[:, 0],
                            np.linalg.norm(connection_obj.anchors_obj.anchor_forces[:, 1:],
                                           axis=1)))
    make_table(sec, "Fastener Demands for Governing Connection", header, units, data,
               alignment='cc', col_formats=formats, width='3in')


def sms_checks(item, sec, sec_title, sub_title, plots_dict, anchors_obj):
    # Extract Parameters
    a = anchors_obj
    if a.anchor_forces.ndim == 3:
        T_lrfd = a.anchor_forces[:, :, 0].max()
        V_lrfd = np.linalg.norm(a.anchor_forces[:, :, 1:], axis=2).max()
        Vx_lrfd = a.anchor_forces[:, :, 1].max()
        Vy_lrfd = a.anchor_forces[:, :, 2].max()
    elif a.anchor_forces.ndim == 2:
        T_lrfd = a.anchor_forces[:, 0].max()
        V_lrfd = np.linalg.norm(a.anchor_forces[:, 1:], axis=1).max()
        Vx_lrfd = a.anchor_forces[:, 1].max()
        Vy_lrfd = a.anchor_forces[:, 2].max()
    else:
        raise ValueError("anchor_forces must be either an (n x t x 3) or (n x 3) array.")
    T_asd = a.results['Tension Demand']
    V_asd = a.results['Shear Demand']
    Vx_asd = a.results['Shear X Demand']
    Vy_asd = a.results['Shear Y Demand']
    T_all = a.results['Tension Capacity']
    Vx_all = a.results['Shear X Capacity']
    Vy_all = a.results['Shear Y Capacity']

    permissible = (a.results['Tension DCR'] != "NG") and \
                  (a.results['Shear X DCR'] != "NG") and \
                  (a.results['Shear Y DCR'] != "NG")

    sec.append(NoEscape(
        "Allowable strengths of sheet metal screws in shear and tension are based upon tabulated values "
        "provided by the California Department of Healcare Access and Information (HCAI) OPD-0001-13: "
        r"\textit{Standard Partition Wall Details.}"))

    with sec.create(Subsubsection('Conversion to ASD Demands')) as ss:
        ss.append("The equipment model was analyzed under ultimate (LRFD) level forces. "
                  "Anchor capacities are tabulated as allowable (ASD) capacities. Maximum anchor tension and "
                  "shear demands are converted by multiplying by the ratio of ASD-factored to LRFD-factored "
                  "lateral loads.")

        with ss.create(Flalign()) as fl:
            fl.append(NoEscape(
                rf'T_{{ASD}} &= T_{{LRFD}}\left(\frac{{ F_{{h,ASD}} }}{{ F_{{h,LRFD}} }}\right) &&= {T_lrfd:.2f}\left(\frac{{ {item.Fah:.2f} }}{{ {item.Fuh:.2f} }}\right)&&={T_lrfd:.2f}({item.Fah / item.Fuh:0.2f}) &&={T_asd:.2f} \text{{ (lbs)}}\\'))
            fl.append(NoEscape(
                rf'V_{{x,ASD}} &= V_{{x,LRFD}}\left(\frac{{ F_{{h,ASD}} }}{{ F_{{h,LRFD}} }}\right) &&= {Vx_lrfd:.2f}\left(\frac{{ {item.Fah:.2f} }}{{ {item.Fuh:.2f} }}\right)&&={Vx_lrfd:.2f}({item.Fah / item.Fuh:0.2f}) &&={Vx_asd:.2f} \text{{ (lbs)}}\\'))
            fl.append(NoEscape(
                rf'V_{{y,ASD}} &= V_{{y,LRFD}}\left(\frac{{ F_{{h,ASD}} }}{{ F_{{h,LRFD}} }}\right) &&= {Vy_lrfd:.2f}\left(\frac{{ {item.Fah:.2f} }}{{ {item.Fuh:.2f} }}\right)&&={Vy_lrfd:.2f}({item.Fah / item.Fuh:0.2f}) &&={Vy_asd:.2f} \text{{ (lbs)}}'''))

    with sec.create(Subsubsection('SMS Capacities from OPD-001-13')) as ss:
        ss.append(
            NoEscape(r'SMS capacities are based on  screw size, base material, and attachment condition.'))

        ss.append(NoEscape(r'\bigskip'))
        ss.append(NewLine())

        with ss.create(MiniPage(width = '4in',pos='t', align='l')) as mini:
            mini.append(NoEscape(r'\begin{footnotesize}'))
            with mini.create(Tabular('ll')) as table:
                table.add_hline()
                table.add_row([NoEscape(r'\rowcolor{lightgray} Properties'), ''])
                table.add_hline()
                table.add_row(['Fastener Size', a.screw_size])
                table.add_hline()
                table.add_row([NoEscape(r'Steel $F_y$ (ksi)'), f'{a.fy:.0f}'])
                table.add_hline()
                table.add_row(['Steel Gauge', f'{a.gauge:.0f}'])
                table.add_hline()
                table.add_row(['Shear X Condition', a.conditions[a.condition_x]['Label']])
                table.add_row(['OPD Table', a.conditions[a.condition_x]['Table']])
                table.add_hline()
                table.add_row(['Shear Y Condition', a.conditions[a.condition_y]['Label']])
                table.add_row(['OPD Table', a.conditions[a.condition_y]['Table']])
                table.add_hline()
            mini.append(NoEscape(r'\end{footnotesize}'))

        ss.append(NoEscape(r'\hfill'))

        with ss.create(MiniPage(width='2.25in',pos='t',align='r')) as mini:
            if not permissible:
                ss.append(NoEscape(
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
                    table.add_row([NoEscape(r'$T_{all}$ (lbs)'), f'{T_all:.0f}'])
                    table.add_row([NoEscape(r'$V_{x,all}$ (lbs)'), f'{Vx_all:.0f}'])
                    table.add_row([NoEscape(r'$V_{y,all}$ (lbs)'), f'{Vy_all:.0f}'])
                    table.add_hline()
                mini.append(NoEscape(r'\end{footnotesize}'))
        ss.append(NoEscape(r'\bigskip'))
        ss.append(NewLine())

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

        if permissible:
            ok_t = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if a.results[
                                                  'Tension DCR'] <= 1 else r'\textcolor{red}{\textbf{\textsf{NG}}}'
            ok_vx = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if a.results[
                                                   'Shear X DCR'] <= 1 else r'\textcolor{red}{\textbf{\textsf{NG}}}'
            ok_vy = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if a.results[
                                                   'Shear Y DCR'] <= 1 else r'\textcolor{red}{\textbf{\textsf{NG}}}'
            ok_v = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if a.results[
                                                  'Shear DCR'] <= 1 else r'\textcolor{red}{\textbf{\textsf{NG}}}'
            ok = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if a.DCR <= 1 else r'\textcolor{red}{\textbf{\textsf{NG}}}'
            ss.append('The resulting demand-to-capacity ratios are:')
            with ss.create(Flalign()) as fl:
                fl.append(
                    rf'DCR_T &= T_{{ASD}} / T_{{all}} &&={T_asd:.0f}/{T_all:.0f} &&={a.results["Tension DCR"]:.2f} &{ok_t}\\')
                fl.append(
                    rf'DCR_{{Vx}} &= V_{{x,ASD}} / V_{{x,all}} &&={Vx_asd:.0f}/{Vx_all:.0f} &&={a.results["Shear X DCR"]:.2f} &{ok_vx}\\')
                fl.append(
                    rf'DCR_{{Vy}} &= V_{{y,ASD}} / V_{{y,all}} &&={Vy_asd:.0f}/{Vy_all:.0f} &&={a.results["Shear Y DCR"]:.2f} &{ok_vy}\\')
                fl.append(
                    rf'DCR_V &= \sqrt{{DCR_{{Vx}}^2 + DCR_{{Vx}}^2}} &&=\sqrt{{ {a.results["Shear X DCR"]:.2f}^2 +{a.results["Shear Y DCR"]:.2f}^2}}&&={a.results["Shear DCR"]:.2f} &{ok_v}\\')
                fl.append(
                    rf'DCR_T &= DCR_T + DCR_V &&= {a.results["Tension DCR"]:.2f} + {a.results["Shear DCR"]:.2f} &&={a.DCR:.2f} &{ok}\\')

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

def concrete_summary_spacing_only(item, sec, sec_title, sub_title, plots_dict, anchor_obj, profile):
    with sec.create(MiniPage(width=r"3.75in", pos='t')) as mini:
        _concrete_input_parameters(mini,anchor_obj,profile)

    _spacing_checks(item, sec, sec_title, sub_title, plots_dict, anchor_obj)


def concrete_summary_full(item, sec, sec_title, sub_title, plots_dict, anchor_obj, profile):
    a = anchor_obj
    # Anchorage Condition Table
    sec.append(NoEscape(rf'''The governing anchor condition consists of ({len(a.xy_group):1}) 
    anchor(s) with loads and details given below. 
    The anchor spacing and edge distances are checked against manufacturer requirements.
    Anchor capacity is evaluated according to the failure mode requirements of ACI 318-19.'''))
    subheader(sec, 'Anchor Data')

    with sec.create(MiniPage(width=NoEscape(r"3.75in"), pos='t', align='l')) as mini:
        # Input Data Table
        _concrete_input_parameters(mini, anchor_obj, profile)

    sec.append(NoEscape(r'\hfill'))
    with sec.create(MiniPage(width=NoEscape(r"2.5in"), pos='t', align='c')) as mini:
        # Anchors Table
        header = [NoEscape(r'\#'),
            NoEscape(r'$x$'),
                  NoEscape(r'$xy$'),
                  NoEscape(r'$T_u$'),
                  NoEscape(r'$V_{ux}$'),
                  NoEscape(r'$V_{uy}$')]
        units = ['', '(in)', '(in)', '(lbs)', '(lbs)', '(lbs)']
        data = []
        for i, (x, y) in enumerate(a.xy_group):

            data.append([a.group_idx[i], x, y, *a.max_group_forces["tension"][i, :]])
        formats = ['{:.0f}', '{:.2f}', '{:.2f}', '{:.0f}', '{:.0f}', '{:.0f}']
        make_table(mini, None, header, units, data, alignment='lccccc',
                   col_formats=formats, use_minipage=False,add_index=False)

        # mini.append(NewLine())
        fig_name = 'diagram'
        width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
        make_figure(mini, width, file, use_minipage=False)





    # Anchor Spacing and Edge Distance Checks
    _spacing_checks(item, sec, sec_title, sub_title, plots_dict, anchor_obj)

    # Tension Limit States Table
    sec.append(NoEscape(r'\begin{samepage}'))
    sec.append(NoEscape(r'\begin{footnotesize}'))
    subheader(sec, 'Tension Failure Modes')
    with sec.create(Tabularx('Xcrrrrrr')) as table:
        table.add_hline()
        header = [NoEscape(r'\rowcolor{lightgray} Limit State'),
                  'Mode',
                  NoEscape('$N_u$'),
                  NoEscape('$N_n$'),
                  NoEscape(r'$\phi$'),
                  NoEscape(r'$\phi_{seismic}$'),
                  NoEscape(r'$\phi\phi_{seismic}N_n$'),
                  'Utilization']
        units = [NoEscape(r'\rowcolor{lightgray}'), '', '(lbs)', '(lbs)', '', '', '(lbs)', '']
        table.add_row(header)
        table.add_row(units)
        table.add_hline()
        results = a.results.loc[a.results['Mode'] == 'Tension']
        for index, row in results.iterrows():
            formatted_row = [f"{item:,.2f}" if isinstance(item, (int, float)) else item for item in
                             row.tolist()]
            formatted_row[-1] = utilization_text_color(formatted_row[-1],row.iloc[-1],1.0)
            table.add_row([index] + formatted_row)
            table.add_hline()
    sec.append(NoEscape(r'\end{footnotesize}'))
    sec.append(NoEscape(r'\end{samepage}'))

    # Shear Limit States Table
    sec.append(NoEscape(r'\begin{samepage}'))
    sec.append(NoEscape(r'\begin{footnotesize}'))
    subheader(sec, 'Shear Limit States')
    with sec.create(Tabularx('Xcrrrrr')) as table:
        table.add_hline()
        header = [NoEscape(r'\rowcolor{lightgray} Limit State'),
                  'Mode',
                  NoEscape('$V_u$'),
                  NoEscape('$V_n$'),
                  NoEscape(r'$\phi$'),
                  NoEscape(r'$\phi V_n$'),
                  'Utilization']
        units = [NoEscape(r'\rowcolor{lightgray}'), '', '(lbs)', '(lbs)', '', '(lbs)', '']
        table.add_row(header)
        table.add_row(units)
        table.add_hline()
        results = a.results.loc[a.results['Mode'] == 'Shear'].drop(columns='Seismic Factor')
        for index, row in results.iterrows():
            formatted_row = [f"{item:,.2f}" if isinstance(item, (int, float)) else item for item in
                             row.tolist()]
            formatted_row[-1] = utilization_text_color(formatted_row[-1], row.iloc[-1], 1.0)
            table.add_row([index] + formatted_row)
            table.add_hline()
    sec.append(NoEscape(r'\end{footnotesize}'))
    sec.append(NoEscape(r'\end{samepage}'))

    # Tension-Shear Interaciton
    sec.append(NoEscape(r'\bigskip'))
    sec.append(NewLine())

    with sec.create(MiniPage(width=r"3.75in", pos='t')) as mini:
        subheader_nobreak(mini,'Tension-Shear Interaction')
        mini.append(NoEscape(rf'''Tension and shear are combined using the interaction criteria provided in 
                            ACI318-19, R17.8.
                            The governing limit states for tension and shear are: 
                            {a.governing_tension_limit} and {a.governing_shear_limit}. 
                            The resulting interaction expression is given below:'''))

        ok = r'\textcolor{Green}{\textbf{\textsf{OK}}}' if a.DCR <= 1 else r'\textcolor{red}{\textbf{\textsf{NG}}}'
        equality = r'\leq' if a.DCR <= 1 else r'>'
        with mini.create(Math(inline=False)) as math:
            math.append(NoEscape(
                rf'\left(\frac{{N_u}}{{\phi N_n}}\right)^{{\frac{{5}}{{3}}}}+\left(\frac{{V_u}}{{\phi V_n}}\right)^{{\frac{{5}}{{3}}}} \\'))
            math.append(NoEscape(
                rf'= \left({a.DCR_N:.2f}\right)^{{\frac{{5}}{{3}}}}+\left({a.DCR_V:.2f}\right)^{{\frac{{5}}{{3}}}} = {a.DCR:.2f}'))
            math.append(NoEscape(rf'{equality} 1 \quad \text{{{ok}}}'))

        # Pull-test Values
        if item.include_pull_test:
            # mini.append(NoEscape(r'\smallskip'))
            mini.append(NewLine())
            subheader(mini, 'Minimum Anchor Pull-Test Value')

            with mini.create(Flalign(numbering=False, escape=False)) as align:
                align.append(rf'''&N_{{\text{{Test}} }}
                    &&=3T_u \geq 500 \text{{ lbs}}  
                    &&= \left(3\right)\left({a.Tu_max:.2f}\right) \geq 500
                    &&={max([3 * a.Tu_max, 500]):.0f} \text{{ lbs}}
                    &\quad \text{{CAC22 (\S6-11.3.2)}}''')


    sec.append(NoEscape(r'\hfill'))
    fig_name = 'interaction'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file)



def _concrete_input_parameters(container, anchor_obj, profile):
    a = anchor_obj
    container.append(NoEscape(r'\begin{footnotesize}'))
    with container.create(Tabularx('lX', pos='t')) as table:
        table.add_hline()
        header = [NoEscape(r'\rowcolor{lightgray} Input Parameters'), '']
        table.add_row(header)
        table.add_hline()
        table.add_row(['Condition', r"Cracked(Assumed)"])
        table.add_hline()
        table.add_row(['Member Profile', rf'{profile}'])
        table.add_hline()
        table.add_row(['Member Thickness (in)', rf'{a.t_slab:.1f}'])
        table.add_hline()
        table.add_row([NoEscape(r"$f'_c$ (psi)"), rf'{a.fc:.0f}'])
        table.add_hline()
        table.add_row([NoEscape(r"$\lambda_a$"), rf'{a.lw_factor_a:.2f}'])
        table.add_hline()
        table.add_row([NoEscape('Anchor Spacing (in), $s_{min}$'), rf'{a.s_min:.2f}'])
        table.add_hline()
        # Use ca values for anchor group, or c values for equipment geometry (if not analysis was run)
        cx_neg = a.cax_neg if a.cax_neg is not None else a.cx_neg
        cx_pos = a.cax_pos if a.cax_pos is not None else a.cx_pos
        cy_neg = a.cay_neg if a.cay_neg is not None else a.cy_neg
        cy_pos = a.cay_pos if a.cay_pos is not None else a.cy_pos
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
        table.add_row(['Anchor Type', rf'{a.anchor_id}'])
        table.add_hline()
        table.add_row(['ESR', rf'{a.esr:.0f}'])
        table.add_hline()
    container.append(NoEscape(r'\end{footnotesize}'))

def _spacing_checks(item, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj
    sec.append(NoEscape(r'\begin{samepage}'))
    subheader(sec, 'Anchor Minimum Spacing and Edge Distances')
    with sec.create(MiniPage(width=r"4in", pos='t')) as mini:
        if a.spacing_requirements['slab_thickness_ok']:
            mini.append(NoEscape(r'''The member thickness meets the minimum required thickness:'''))
            with mini.create(Math(inline=False)) as m:
                m.append(NoEscape(r't_{slab} \geq h_{min} \rightarrow '))
                m.append(NoEscape(
                    rf'{a.t_slab:.2f} \geq {a.hmin1:.2f} \quad \text{{\textcolor{{Green}}{{ \textbf{{\textsf{{OK}} }} }} }}'))
        else:
            mini.append(NoEscape(r'''The member thickness is insufficient for the chosen anchor:'''))
            with mini.create(Math(inline=False)) as m:
                m.append(NoEscape(rf't_{{slab}} = {a.t_slab:.2f} < h_{{min}} \rightarrow '))
                m.append(NoEscape(
                    rf'{a.hmin1:.2f} \quad \text{{ \textcolor{{red}}{{\textbf{{\textsf{{NG}} }} }} }}'))


        mini.append(NoEscape(rf'''The acceptance criteria for spacing and edge distance provided by ESR-{a.esr:.0f} 
        are listed below corresponding to the given concrete thickness. The plot to the right shows the given anchor spacing and edge distances 
        against these criteria \\'''))
        with mini.create(Alignat(numbering=False, escape=False)) as m:
            m.append(rf'c_{{min}} = {a.c1:.2f} \text{{ in}} \text{{ for }} s \geq {a.s1:.2f} \text{{ in,}} \quad')
            # m.append(rf'\text{{for }}s\geq {a.s1:.2f} \text{{ in}}\\')
            m.append(rf's_{{min}} = {a.s2:.2f} \text{{ in}} \text{{ for }}c\geq {a.c2:.2f} \text{{ in}}\\')
            # m.append(rf'\text{{for }}c\geq {a.c2:.2f} \text{{ in}}\\')
        if a.spacing_requirements['edge_and_spacing_ok']:
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

    if not all(a.spacing_requirements.values()):
        sec.append(NoEscape(r'\bigskip'))
        sec.append(NewLine())
        sec.append(NoEscape(
            r'\textcolor{red}{\textbf{Anchor does not meet member thickness and/or spacing requirements!}}'))
    sec.append(NoEscape(r'\end{samepage}'))

def anchor_tension(_, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj
    if a.Nsa:
        sec.append(NoEscape(rf'Tensile strength reported by manufacturer\
                            (see ESR {a.esr:.0f}): $N_{{sa}} = {a.Nsa:.0f}$ lbs'))
        # with sec.create(Flalign()) as fl:
        #     fl.append(NoEscape(
        #         rf'&N_{{sa}} = {a.Nsa} \text{{ lbs}} && \text{{Manufacturer-Provided Value. See ESR: {a.esr:.0f}}}'))
    else:
        # Todo: provide report option for non-tabulated Nsa
        raise Exception('Need to provide report functionality for non-tabulated Nsa')


def tension_breakout(_, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj


    with sec.create(MiniPage(width=NoEscape('2.5in'),pos='t')) as leftcol:
        # Input Paramters Table
        leftcol.append(NoEscape(r'\begin{footnotesize}'))
        with leftcol.create(Tabularx('Xcc', pos='t')) as table:
            table.add_hline()
            header = [NoEscape(r'\rowcolor{lightgray} Input Parameters'), '', '']
            table.add_row(header)
            table.add_hline()
            table.add_row(['Effective embedment (in)', NoEscape('$h_{ef}$'), NoEscape(f'{a.hef}')])
            table.add_hline()
            table.add_row(['Edge Distances (in)', NoEscape('$c_{a,x+}$'), NoEscape(f'{a.cax_pos:.2f}')])
            table.add_row(['', NoEscape('$c_{a,x-}$'), NoEscape(f'{a.cax_neg:.2f}')])
            table.add_row(['', NoEscape('$c_{a,y+}$'), NoEscape(f'{a.cay_pos:.2f}')])
            table.add_row(['', NoEscape('$c_{a,y-}$'), NoEscape(f'{a.cay_neg:.2f}')])
            table.add_hline()
            table.add_row(['Critical Edge Dist. (in)', NoEscape('$c_{ac}$'), NoEscape(f'{a.cac:.2f}')])
            table.add_hline()
            table.add_row(['Effectivness Factor', NoEscape('$k_c$'), NoEscape(f'{a.kc:.1f}')])
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
                rf'=9({a.hef:.1f})^2 &&={a.Anco:.1f} \text{{ in}}^2 &&\text{{ACI318-19 (17.6.2.1.4)}}\\'))

            # ANc
            fl.append(NoEscape(rf'''&A_{{Nc}} &&=({a.bxN:.1f})\times ({a.byN:.1f}) 
            && = {a.Anc:.1f} \text{{ in}}^2 && \text{{ACI318-19 17.6.2.1.2}}\\ '''))

            # Psi,ecN (Eccentricity Factor)
            fl.append(rf'''&e_{{Nx}} &&= \frac{{\sum{{ \left(x_i-\bar{{x}}\right) T_i }}}}{{ \sum{{T_i}} }}
                &&={a.ex:.2f} \text{{ in}}\\''')
            fl.append(rf'''&\psi_{{ec,Nx}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{Nx}} }}{{1.5h_{{ef}} }}\right)}}
            && ={a.psi_ecNx:.2f}\text{{ in}} && \text{{ACI318-19 (17.6.2.3.1)}}\\''')

            fl.append(rf'''&e_{{Ny}} &&= \frac{{\sum{{ \left(y_i-\bar{{y}}\right) T_i }}}}{{ \sum{{T_i}} }}
                &&={a.ey:.2f} \text{{ in}}\\''')
            fl.append(rf'''&\psi_{{ec,Ny}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{Ny}} }}{{1.5h_{{ef}} }}\right)}}
            && ={a.psi_ecNy:.2f}\text{{ in}} && \text{{ACI318-19 (17.6.2.3.1)}}\\''')

            fl.append(rf'''&\psi_{{ec,N}} && =\psi_{{ec,Nx}} \times \psi_{{ec,Ny}} &&={a.psi_ecN:.2f}\\''')

            # Psi,ed (Edge Factor)
            fl.append(rf'''&\psi_{{ed,N}} && = \min{{\left(1.0,\quad 0.7 + 0.3\frac {{ c_{{a,min}}}}{{1.5h_{{ef}} }} \right)}}
                &&={a.psi_edN:.2f} &&\text{{ACI318-19 (17.6.2.4.1)}}\\''')

            # Psi,cN (Breakout cracking factor)
            if a.psi_cN:
                fl.append(rf'''&\psi_{{c,N}} &&  
                &&={a.psi_cN:.2f} && \text{{ESR: {a.esr:.0f}}}\\''')
            else:
                # Todo: provide report option for non-tabulated Breakout cracking factor
                raise Exception('Need to provide report functionality for non-tabulated psi_cN')

            # Psi,cpN (Breakout splitting factor)
            fl.append(
                r'&\psi_{cp,N} && = \min{\left(1.0, \max{\left(\frac{c_{a,min}}{c_{ac}},\frac{1.5h_{ef}}{c_{ac}}\right)}\right)}')
            fl.append(rf' &&={a.psi_cpN:.2f} &&\text{{ACI318-19 (17.6.2.6.1)}}\\')

            # Nb
            if all([a.n_anchor == 1,
                    a.anchor_type in ['Headed Stud', 'Headed Bolt'],
                    11 <= a.hef <= 25]):
                fl.append(rf'''&N_b && =16\lambda_a\sqrt{{f\prime _c}} h_{{ef}}^{{5/3}}
                && =16({a.lw_factor_a})\sqrt{{ ({a.fc}) }}({a.hef})^{{5/3}}
                ={a.Nb:.2f} \text{{ lbs}} && \text{{ACI318-19 (17.6.2.2.3)}}\\''')
            else:
                fl.append(rf'''&N_b && =k_c\lambda_a\sqrt{{f_c^\prime}} h_{{ef}}^{{1.5}}
                            =({a.kc})({a.lw_factor_a})\sqrt{{ ({a.fc}) }}({a.hef})^{{1.5}}
                            &&={a.Nb:.2f} \text{{ lbs}} && \text{{ACI318-19 (17.6.2.2.1)}}\\''')

            # Ncb
            fl.append(r'&N_{cb} &&= \frac{A_{Nc}}{A_{Nco}}\psi_{ec,N}\psi_{ed,N}\psi_{c,N}\psi_{cp,N}N_b')
            fl.append(rf'&&={a.Ncb:.2f} \text{{ lbs}} && \text{{ACI318-19 (17.6.2.1)}}')
        rightcol.append(NoEscape(r'}')) # Close local font size group


def tension_breakout_OLD(_, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj

    # Breakout Figure
    # file, width = plots._anchor_diagram_vtk(a, show_tension_breakout=True,
    #                                         show_tension_forces=True)
    fig_name = 'diagram'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file)

    # Input Parameters Table
    sec.append(NoEscape(r'\hfill'))
    with sec.create(MiniPage(width=r'3.75in', pos='t')) as mini:
        with mini.create(Tabularx('Xcc', pos='t')) as table:
            table.add_hline()
            header = [NoEscape(r'\rowcolor{lightgray} Input Parameters'), '', '']
            table.add_row(header)
            table.add_hline()
            table.add_row(['Effective embedment (in)', NoEscape('$h_{ef}$'), NoEscape(f'{a.hef}')])
            table.add_hline()
            table.add_row(['Edge Distances (in)', NoEscape('$c_{a,x+}$'), NoEscape(f'{a.cax_pos:.2f}')])
            table.add_row(['', NoEscape('$c_{a,x-}$'), NoEscape(f'{a.cax_neg:.2f}')])
            table.add_row(['', NoEscape('$c_{a,y+}$'), NoEscape(f'{a.cay_pos:.2f}')])
            table.add_row(['', NoEscape('$c_{a,y-}$'), NoEscape(f'{a.cay_neg:.2f}')])
            table.add_hline()
            table.add_row(['Critical Edge Dist. (in)', NoEscape('$c_{ac}$'), NoEscape(f'{a.cac:.2f}')])
            table.add_hline()
            table.add_row(['Effectivness Factor', NoEscape('$k_c$'), NoEscape(f'{a.kc:.1f}')])
            table.add_hline()
    sec.append(NewLine())
    sec.append(NoEscape(r'\smallskip'))
    with sec.create(Flalign()) as fl:
        # ANco
        fl.append(NoEscape(r'&A_{Nco} &&= 9h_{ef}^2'))
        fl.append(NoEscape(
            rf'=9({a.hef:.1f})^2 &&={a.Anco:.1f} \text{{ in}}^2 &&\text{{ACI318-19 (17.6.2.1.4)}}\\'))

        # ANc
        fl.append(NoEscape(rf'''&A_{{Nc}} &&=({a.bxN:.1f})\times ({a.byN:.1f}) 
        && = {a.Anc:.1f} \text{{ in}}^2 && \text{{ACI318-19 17.6.2.1.2}}\\ '''))

        # Psi,ecN (Eccentricity Factor)
        fl.append(rf'''&e_{{Nx}} &&= \frac{{\sum{{ \left(x_i-\bar{{x}}\right) T_i }}}}{{ \sum{{T_i}} }}
            &&={a.ex:.2f} \text{{ in}}\\''')
        fl.append(rf'''&\psi_{{ec,Nx}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{Nx}} }}{{1.5h_{{ef}} }}\right)}}
        && ={a.psi_ecNx:.2f}\text{{ in}} && \text{{ACI318-19 (17.6.2.3.1)}}\\''')

        fl.append(rf'''&e_{{Ny}} &&= \frac{{\sum{{ \left(y_i-\bar{{y}}\right) T_i }}}}{{ \sum{{T_i}} }}
            &&={a.ey:.2f} \text{{ in}}\\''')
        fl.append(rf'''&\psi_{{ec,Ny}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{Ny}} }}{{1.5h_{{ef}} }}\right)}}
        && ={a.psi_ecNy:.2f}\text{{ in}} && \text{{ACI318-19 (17.6.2.3.1)}}\\''')

        fl.append(rf'''&\psi_{{ec,N}} && =\psi_{{ec,Nx}} \times \psi_{{ec,Ny}} &&={a.psi_ecN:.2f}\\''')

        # Psi,ed (Edge Factor)
        fl.append(rf'''&\psi_{{ed,N}} && = \min{{\left(1.0,\quad 0.7 + 0.3\frac {{ c_{{a,min}}}}{{1.5h_{{ef}} }} \right)}}
            &&={a.psi_edN:.2f} &&\text{{ACI318-19 (17.6.2.4.1)}}\\''')

        # Psi,cN (Breakout cracking factor)
        if a.psi_cN:
            fl.append(rf'''&\psi_{{c,N}} &&  
            &&={a.psi_cN:.2f} && \text{{ESR: {a.esr:.0f}}}\\''')
        else:
            # Todo: provide report option for non-tabulated Breakout cracking factor
            raise Exception('Need to provide report functionality for non-tabulated psi_cN')

        # Psi,cpN (Breakout splitting factor)
        fl.append(
            r'&\psi_{cp,N} && = \min{\left(1.0, \max{\left(\frac{c_{a,min}}{c_{ac}},\frac{1.5h_{ef}}{c_{ac}}\right)}\right)}')
        fl.append(rf' &&={a.psi_cpN:.2f} &&\text{{ACI318-19 (17.6.2.6.1)}}\\')

        # Nb
        if all([a.n_anchor == 1,
                a.anchor_type in ['Headed Stud', 'Headed Bolt'],
                11 <= a.hef <= 25]):
            fl.append(rf'''&N_b && =16\lambda_a\sqrt{{f\prime _c}} h_{{ef}}^{{5/3}}
            && =16({a.lw_factor_a})\sqrt{{ ({a.fc}) }}({a.hef})^{{5/3}}
            ={a.Nb:.2f} \text{{ lbs}} && \text{{ACI318-19 (17.6.2.2.3)}}\\''')
        else:
            fl.append(rf'''&N_b && =k_c\lambda_a\sqrt{{f_c^\prime}} h_{{ef}}^{{1.5}}
                        =({a.kc})({a.lw_factor_a})\sqrt{{ ({a.fc}) }}({a.hef})^{{1.5}}
                        &&={a.Nb:.2f} \text{{ lbs}} && \text{{ACI318-19 (17.6.2.2.1)}}\\''')

        # Ncb
        fl.append(r'&N_{cb} &&= \frac{A_{Nc}}{A_{Nco}}\psi_{ec,N}\psi_{ed,N}\psi_{c,N}\psi_{cp,N}N_b')
        fl.append(rf'&&={a.Ncb:.2f} \text{{ lbs}} && \text{{ACI318-19 (17.6.2.1)}}')


def tension_pullout(_, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj
    if a.Np:
        sec.append(NoEscape(rf'Pull-out strength reported by manufacturer\
                            (see ESR {a.esr:.0f}): $N_{{p}} = {a.Np:.0f}$ lbs'))
        # with sec.create(Flalign()) as fl:
        #     fl.append(NoEscape(
        #         rf'&N_{{sa}} = {a.Nsa} \text{{ lbs}} && \text{{Manufacturer-Provided Value. See ESR: {a.esr:.0f}}}'))
    else:
        # Todo: provide report option for non-tabulated Np
        raise Exception('Need to provide report functionality for non-tabulated Np')


def side_face_blowout(_, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj
    # todo: add side face blowout section (for headed studs)


def bond_strength(_, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj
    # todo: add bond strength section (for epoxy base_anchors)


def anchor_shear(_, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj
    if a.Vsa:
        sec.append(NoEscape(rf'Anchor shear strength reported by manufacturer\
                            (see ESR {a.esr:.0f}): $V_{{sa}} = {a.Vsa:.0f}$ lbs'))
    else:
        # Todo: provide report option for non-tabulated Vsa
        raise Exception('Need to provide report functionality for non-tabulated Vsa')


def shear_breakout(_, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj
    key = a.governing_shear_breakout_case

    with sec.create(MiniPage(width=NoEscape('2.5in'), pos='t')) as leftcol:
        # Input Parameter Table
        leftcol.append(NoEscape(r'\begin{footnotesize}'))
        with leftcol.create(Tabularx('Xcc', pos='t')) as table:
            table.add_hline()
            table.add_row([NoEscape(r'\rowcolor{lightgray} Input Parameters'), '', ''])
            table.add_hline()
            table.add_row(['Effective embedment (in)', NoEscape('$h_{ef}$'), NoEscape(f'{a.hef}')])
            table.add_hline()
            table.add_row(['Edge Distances (in)',
                           NoEscape('$c_{a1}$'), NoEscape(f'{a.vcb_pars[key]["ca1"]}')])
            table.add_row(['', NoEscape('$c_{a2+}$'), NoEscape(f'{a.vcb_pars[key]["ca2+"]}')])
            table.add_row(['', NoEscape('$c_{a2-}$'), NoEscape(f'{a.vcb_pars[key]["ca2-"]}')])
            table.add_hline()
            table.add_row(['Load Bearing Length (in)', NoEscape('$l_{e}$'), NoEscape(f'{a.le}')])
            table.add_hline()
            table.add_row(['Anchor Diameter (in)', NoEscape('$d_a$'), NoEscape(f'{a.da}')])
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
            fl.append(rf'''=4.5({a.vcb_pars[key]["ca1"]})^2 &&={a.vcb_pars[key]["Avco"]} \text{{ in}}^2 
                             &&\text{{ACI318-19 (17.7.2.1.3)}}\\''')

            # AVc
            fl.append(rf'''&A_{{Vc}} &&=({a.vcb_pars[key]["b"]})\times ({a.vcb_pars[key]["ha"]}) 
                            && = {a.vcb_pars[key]["Avc"]} \text{{ in}}^2 && \text{{ACI318-19 17.7.2.1.1}}\\ ''')

            # Psi,ecN (Eccentricity Factor)
            if a.governing_shear_breakout_case[0] == 'y':
                fl.append(rf'''&e_{{V}} &&= \frac{{\sum{{ \left(x_i-\bar{{x}}\right) V_{{yi}} }}}}{{ \sum{{V_{{yi}}}} }}
                                    &&={a.vcb_pars[key]['eV']:.2f} \text{{ in}}\\''')
                fl.append(rf'''&\psi_{{ec,V}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{V}} }}{{1.5c_{{a1}} }}\right)}}
                                && ={a.vcb_pars[key]['psi_ecV']:.2f}\text{{ in}} && \text{{ACI318-19 (17.7.2.3.1)}}\\''')
            else:
                fl.append(rf'''&e_{{V}} &&= \frac{{\sum{{ \left(y_i-\bar{{y}}\right) V_{{yi}} }}}}{{ \sum{{V_{{yi}}}} }}
                                    &&={a.vcb_pars[key]['eV']} \text{{ in}}\\''')
                fl.append(rf'''&\psi_{{ec,V}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{V}} }}{{1.5c_{{a1}} }}\right)}}
                                && ={a.vcb_pars[key]['psi_ecV']:.2f}\text{{ in}} && \text{{ACI318-19 (17.7.2.3.1)}}\\''')

            # Psi,ed (Edge Factor)
            fl.append(rf'''&\psi_{{ed,V}} && = \min{{\left(1.0,\quad 0.7 + 0.3\frac {{ c_{{a2,min}}}}{{1.5h_{{ef}} }} \right)}}
                                &&={a.vcb_pars[key]['psi_edV']:.2f} &&\text{{ACI318-19 (17.7.2.4.1)}}\\''')

            # Psi,cV (Breakout cracking factor)
            fl.append(rf'''&\psi_{{c,V}} &&
                            &&={a.vcb_pars[key]["psi_cV"]:.2f} &&\text{{ACI318-19 17.7.2.5.1}}\\''')

            # Psi,hV (Breakout thickenss factor)
            fl.append(
                r'&\psi_{h,V} && = \max{\left(1.0,\quad \sqrt{\frac{1.5c_{a1}}{h_{a}}}\right)}')
            fl.append(rf' &&={a.vcb_pars[key]["psi_hV"]:.2f} &&\text{{ACI318-19 (17.7.2.6.1)}}\\')

            # Vb
            fl.append(r'''&V_b &&= \min{ \left( 7 \left(\frac{l_e}{d_a}\right)^{0.2}\sqrt{d_a},\quad 9\right)}
                            \lambda_a \sqrt{f_c'}\left(c_{a1}\right)^{1.5}''')
            fl.append(
                rf'&& = {a.vcb_pars[key]["Vb"]:,.2f} \text{{ lbs}} && \text{{ACI318-19 (17.7.2.2.1)}}\\')

            # # Vcb
            fl.append(r'&V_{cb} &&= \frac{A_{Vc}}{A_{Vco}}\psi_{ec,V}\psi_{ed,V}\psi_{c,V}\psi_{h,V}V_b')
            fl.append(
                rf'&&={a.vcb_pars[key]["Vcb"]:,.2f} \text{{ lbs}} && \text{{ACI318-19 (17.7.2.1)}}')
        rightcol.append(NoEscape(r'}')) # Close local font size group
    sec.append(NewLine())
    breakout_text = {'xp_edge': 'positive $X$ face breakout of edge anchors',
                     'xp_full': 'positive $X$ face breakout of anchor group',
                     'xn_edge': 'negative $X$ face breakout of edge anchors',
                     'xn_full': 'negative $X$ face breakout of anchors group',
                     'yp_edge': 'positive $Y$ face breakout of edge anchors',
                     'yp_full': 'positive $Y$ face breakout of anchor group',
                     'yn_edge': 'negative $Y$ face breakout of edge anchors',
                     'yn_full': 'negative $Y$ face breakout of anchor group'}

    sec.append(NoEscape(r'{\footnotesize'))
    sec.append(NoEscape('''Note: ACI318 specifies that unless base anchors are welded to a common base plate,
        two conditions should be checked:'''))
    with sec.create(Enumerate()) as enum:
        enum.add_item('Full shear breakout cone against total load to all base anchors')
        enum.add_item('Breakout cone of edge base anchors against load to those base anchors.')

    sec.append(NoEscape(f'''The two breakout modes (edge anchors only, and full anchor group) are 
        checked in positive and negative $X$- and $Y$-directions where an edge condition is present.
        The results from all applicable breakout cases are provided in the anchorage summary section.
        Here, calculations are provided for the governing breakout of the 
        {breakout_text[key]}. '''))
    sec.append(NoEscape(r'}'))


def shear_breakout_OLD(_, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj

    breakout_text = {'xp_edge': 'positive $X$ face breakout of edge anchors',
                     'xp_full': 'positive $X$ face breakout of anchor group',
                     'xn_edge': 'negative $X$ face breakout of edge anchors',
                     'xn_full': 'negative $X$ face breakout of anchors group',
                     'yp_edge': 'positive $Y$ face breakout of edge anchors',
                     'yp_full': 'positive $Y$ face breakout of anchor group',
                     'yn_edge': 'negative $Y$ face breakout of edge anchors',
                     'yn_full': 'negative $Y$ face breakout of anchor group'}

    key = a.governing_shear_breakout_case

    sec.append(NoEscape('''ACI318 spefies that unless base anchors are welded to a common base plate,
    two conditions should be checked:'''))
    with sec.create(Enumerate()) as enum:
        enum.add_item('Full shear breakout cone against total load to all base anchors')
        enum.add_item('Breakout cone of edge base anchors against load to those base anchors.')

    sec.append(NoEscape(f'''The two breakout modes (edge anchors only, and full anchor group) are 
    checked in positive and negative $X$- and $Y$-directions where an edge condition is present.
    The results from all applicable breakout cases are provided in the anchorage summary section.
    Here, calculations are provided for the governing breakout of the 
    {breakout_text[key]}. '''))
    sec.append(NewLine())

    # Breakout Cone Graphic
    # file, width = plots._anchor_diagram_vtk(a, shear_breakout_case=a.governing_shear_breakout_case)
    fig_name = 'diagram'
    width, file = plots_dict[make_figure_filename(sec_title, sub_title, fig_name)]
    make_figure(sec, width, file)

    # Input Parameter Table
    sec.append(NoEscape(r'\hfill'))
    with sec.create(MiniPage(width=r'3.75in', pos='t')) as mini:
        with mini.create(Tabularx('Xcc', pos='t')) as table:
            table.add_hline()
            table.add_row([NoEscape(r'\rowcolor{lightgray} Input Parameters'), '', ''])
            table.add_hline()
            table.add_row(['Effective embedment (in)', NoEscape('$h_{ef}$'), NoEscape(f'{a.hef}')])
            table.add_hline()
            table.add_row(['Edge Distances (in)',
                           NoEscape('$c_{a1}$'), NoEscape(f'{a.vcb_pars[key]["ca1"]:.2f}')])
            table.add_row(['', NoEscape('$c_{a2+}$'), NoEscape(f'{a.vcb_pars[key]["ca2+"]:.2f}')])
            table.add_row(['', NoEscape('$c_{a2-}$'), NoEscape(f'{a.vcb_pars[key]["ca2-"]:.2f}')])
            table.add_hline()
            table.add_row(['Load Bearing Length (in)', NoEscape('$l_{e}$'), NoEscape(f'{a.le:.2f}')])
            table.add_hline()
            table.add_row(['Anchor Diameter (in)', NoEscape('$d_a$'), NoEscape(f'{a.da:.2f}')])
            table.add_hline()
    sec.append(NewLine())
    sec.append(NoEscape(r'\smallskip'))

    with sec.create(Flalign()) as fl:
        # AVco
        fl.append(r'&A_{Vco} &&= 4.5c_{a1}^2')
        fl.append(rf'''=4.5({a.vcb_pars[key]["ca1"]})^2 &&={a.vcb_pars[key]["Avco"]} \text{{ in}}^2 
                         &&\text{{ACI318-19 (17.7.2.1.3)}}\\''')

        # AVc
        fl.append(rf'''&A_{{Vc}} &&=({a.vcb_pars[key]["b"]})\times ({a.vcb_pars[key]["ha"]}) 
                        && = {a.vcb_pars[key]["Avc"]} \text{{ in}}^2 && \text{{ACI318-19 17.7.2.1.1}}\\ ''')

        # Psi,ecN (Eccentricity Factor)
        if a.governing_shear_breakout_case[0] == 'y':
            fl.append(rf'''&e_{{V}} &&= \frac{{\sum{{ \left(x_i-\bar{{x}}\right) V_{{yi}} }}}}{{ \sum{{V_{{yi}}}} }}
                                &&={a.vcb_pars[key]['eV']:.2f} \text{{ in}}\\''')
            fl.append(rf'''&\psi_{{ec,V}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{V}} }}{{1.5c_{{a1}} }}\right)}}
                            && ={a.vcb_pars[key]['psi_ecV']:.2f}\text{{ in}} && \text{{ACI318-19 (17.7.2.3.1)}}\\''')
        else:
            fl.append(rf'''&e_{{V}} &&= \frac{{\sum{{ \left(y_i-\bar{{y}}\right) V_{{yi}} }}}}{{ \sum{{V_{{yi}}}} }}
                                &&={a.vcb_pars[key]['eV']} \text{{ in}}\\''')
            fl.append(rf'''&\psi_{{ec,V}} &&= \frac{{1}}{{ \left(1+\frac{{ e_{{V}} }}{{1.5c_{{a1}} }}\right)}}
                            && ={a.vcb_pars[key]['psi_ecV']:.2f}\text{{ in}} && \text{{ACI318-19 (17.7.2.3.1)}}\\''')

        # Psi,ed (Edge Factor)
        fl.append(rf'''&\psi_{{ed,V}} && = \min{{\left(1.0,\quad 0.7 + 0.3\frac {{ c_{{a2,min}}}}{{1.5h_{{ef}} }} \right)}}
                            &&={a.vcb_pars[key]['psi_edV']:.2f} &&\text{{ACI318-19 (17.7.2.4.1)}}\\''')

        # Psi,cV (Breakout cracking factor)
        fl.append(rf'''&\psi_{{c,V}} &&
                        &&={a.vcb_pars[key]["psi_cV"]:.2f} &&\text{{ACI318-19 17.7.2.5.1}}\\''')

        # Psi,hV (Breakout thickenss factor)
        fl.append(
            r'&\psi_{h,V} && = \max{\left(1.0,\quad \sqrt{\frac{1.5c_{a1}}{h_{a}}}\right)}')
        fl.append(rf' &&={a.vcb_pars[key]["psi_hV"]:.2f} &&\text{{ACI318-19 (17.7.2.6.1)}}\\')

        # Vb
        fl.append(r'''&V_b &&= \min{ \left( 7 \left(\frac{l_e}{d_a}\right)^{0.2}\sqrt{d_a},\quad 9\right)}
                        \lambda_a \sqrt{f_c'}\left(c_{a1}\right)^{1.5}''')
        fl.append(
            rf'&& = {a.vcb_pars[key]["Vb"]:,.2f} \text{{ lbs}} && \text{{ACI318-19 (17.7.2.2.1)}}\\')

        # # Vcb
        fl.append(r'&V_{cb} &&= \frac{A_{Vc}}{A_{Vco}}\psi_{ec,V}\psi_{ed,V}\psi_{c,V}\psi_{h,V}V_b')
        fl.append(
            rf'&&={a.vcb_pars[key]["Vcb"]:,.2f} \text{{ lbs}} && \text{{ACI318-19 (17.7.2.1)}}')


def shear_pryout(_, sec, sec_title, sub_title, plots_dict, anchor_obj):
    a = anchor_obj
    with sec.create(Flalign()) as fl:
        fl.append(r'& k_{cp} &&= \begin{cases} '
                  r'1.0 & \text{for } h_{ef} < 2.5 \text{ in}\\'
                  r'2.0 & \text{for } h_{ef} \geq 2.5 \text{ in} '
                  r'\end{cases}')
        fl.append(rf'&&={a.kcp}\\')
        fl.append(r'& V_{cp} && = k_{cp}N_{cp}')
        fl.append(rf'&&={a.Vcp:,.2f}\text{{ lbs}} &&\text{{ACI318-19 (17.7.3.1)}}')


def cmu_summary_full(item, sec, sec_title, sub_title, plots_dict):
    sec.append('See attached calculations for anchorage checks to CMU.')


def model_instability(item, sec, sec_title, sub_title, plots_dict):
    sec.append(NoEscape(
        r'\textcolor{red}{\textbf{\textsf{AN INSTABILITY EXISTS IN THE STRUCTURAL MODEL. REVIEW MODEL GEOMETRY, MATERIAL DEFINITIONS, AND FORCE RELEASES FOR MODEL COMPLETNESS.}}}'))


EquipmentReportSections.initialize_plots_list()
