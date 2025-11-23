import xlwings as xw
import pandas as pd
from anchor_pro.elements.sms import SMSCatalog

class ExcelTablesImporter:
    NA_VALUES = ['NA', 'N/A', '-', 'None', ""]

    def __init__(self, excel_path):
        print('Reading data from Excel')
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
        self.df_equipment = None
        self.df_base_geometry = None
        self.df_wall_geometry = None
        self.df_fasteners = None
        self.df_concrete = None
        self.df_walls = None
        self.df_anchors = None
        self.df_brackets_catalog = None
        self.bracket_group_list = None
        self.df_sms = None
        self.df_product_groups = None
        self.df_wood = None


        # Table References (sheet name, table NW cell name, instance attribute name)
        table_references = [('Equipment', 'tbl_equipment', 'df_equipment'),
                            ('Base Geometry', 'tbl_base_geometry', 'df_base_geometry'),
                            ('Wall Geometry', 'tblBrackets', 'df_wall_geometry'),
                            ('Fastener Patterns', 'tblBacking', 'df_fasteners'),
                            ('Concrete', 'tbl_concrete', 'df_concrete'),
                            ('Wood','tbl_wood', 'df_wood'),
                            ('Walls', 'tblWalls', 'df_walls'),
                            ('Anchors', 'tbl_anchors', 'df_anchors'),
                            ('Wood Fasteners', 'tbl_wood_fasteners', 'df_wood_fasteners'),
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

        # Adjust Indices on Certain Tables
        for df in [self.df_walls, self.df_fasteners]:
            df.set_index(df.columns[0], inplace=True)

        # Import Bracket Product Group Names
        self.bracket_group_list = wb.names['bracket_groups_list'].refers_to_range.value

        #Create Special SMS Catalog Opbject
        self.sms_catalog = SMSCatalog(self.df_sms)