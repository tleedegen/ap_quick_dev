import xlwings as xw
import pandas as pd

class ExcelTablesImporter:
    NA_VALUES = ['NA', 'N/A', '-', 'None', ""]

    def __init__(self, excel_path):
        print(f'Reading data from Excel')
        self.path = excel_path
        wb = xw.Book(self.path)

        # Import Project Info from Named Cells on Project Worksheet
        project_sheet = wb.sheets['Project']
        self.project_info = {}
        for name in wb.names:
            if name.name.startswith('_xlfn') or name.name.startswith('_xlpm') or name.name.startswith('_xleta'):
                continue
            try:
                if name.refers_to_range.sheet == project_sheet:
                    value = name.refers_to_range.value
                    self.project_info[name.name] = value
            except Exception as e:
                print(f"Skipping {name.name} due to error: {e}")

        # Import Excel Tables
        self.df_anchor_designs = None
        self.df_base_materials = None
        self.df_anchors_concrete = None
        self.df_anchors_cmu = None
        self.df_anchors_epoxy = None
        self.df_product_groups = None

        # Table References (instance attribute name, table NW cell name, sheet name)
        table_references = [('df_anchor_designs', 'Design', 'tblDesigns'),
                            ('Base Geometry', 'tbl_base_geometry', 'df_base_geometry'),
                            ('Wall Geometry', 'tblBrackets', 'df_wall_geometry'),
                            ('Fastener Patterns', 'tblBacking', 'df_fasteners'),
                            ('Concrete', 'tbl_concrete', 'df_concrete'),
                            ('Walls', 'tblWalls', 'df_walls'),
                            ('Anchors', 'tbl_anchors', 'df_anchors'),
                            ('SMS', 'tblSMS', 'df_sms'),
                            ('Brackets', 'tblBracketCatalog', 'df_brackets_catalog'),
                            ('Anchor Product Groups', 'tblProductGroups', 'df_product_groups')]

        for (sheet_name, tbl_name, df_name) in table_references:
            sheet = wb.sheets[sheet_name]
            table_cell = sheet.range(tbl_name)
            start_address = table_cell.get_address(0, 0, include_sheetname=True, external=False)
            df = sheet.range(start_address).expand().options(pd.DataFrame,
                                                             header=1,
                                                             index=False,
                                                             expand='table').value
            setattr(self, df_name, df)

        # Replace undesired inputs
        self.df_equipment = self.df_equipment.replace(ExcelTablesImporter.NA_VALUES, None)

        # Adjust Indicies on Certain Tables
        for df in [self.df_walls, self.df_product_groups, self.df_fasteners]:
            df.set_index(df.columns[0], inplace=True)

        # Import Bracket Product Group Names
        self.bracket_group_list = wb.names['bracket_groups_list'].refers_to_range.value
