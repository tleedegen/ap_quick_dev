from anchor_pro.concrete_anchors import ConcreteAnchors
import numpy as np
import xlwings as xw
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import anchor_pro.plots as plts
import vtk

'''Generic Inputs'''
xlpath = r"C:\Users\djmiller\OneDrive - Degenkolb Engineers\AnchorPro\Reference Files\Mechanical Anchor Catalog.xlsx"

''' Data Which Should Come from the Frontend'''
xy_anchors = [[-5, -5], [5, -5], [5, 5], [-5, 5]]
anchor_forces = np.array([[[100,50,50]],[[100,50,50]],[[100,50,50]],[[100,50,50]]])  # Array with dims (n_anchors, 1, 3). Last index is (N, Vx, Vy)

concrete_data = pd.Series({'Bx': 10, # Bx and By could be user-input or simply calculated as the bounding box around the anchors
                 'By': 10,
                 'fc': 4000,
                 'lw_factor': 1,
                 'cracked_concrete': True,
                 'poisson': 0.25,
                 't_slab': 6,
                 'cx_neg': 6,
                 'cx_pos': np.inf,
                 'cy_neg': np.inf,
                 'cy_pos': np.nan,
                 'profile': 'Slab',  # Slab or Filled Deck
                 'anchor_position': 'top',  #top or soffit
                 'weight_classification_base': 'NWC'  # Either NWC or LWC
})

anchor_id = 'HILTI KBTZ2-C [0.375X1.5]'

'''Data from the anchor catalog'''
wb = xw.Book(xlpath)
sheet = wb.sheets['Anchors']
table_cell = sheet.range('tbl_anchors')
start_address = table_cell.get_address(0, 0, include_sheetname=True, external=False)
df = sheet.range(start_address).expand().options(pd.DataFrame,
                                                 header=1,
                                                 index=False,
                                                 expand='table').value

anchor_data = df[df['anchor_id']==anchor_id].iloc[0]

def main():
    # Create Object and Set Properties
    print('Creating anchors object and setting properties')
    model = ConcreteAnchors()
    model.set_data(concrete_data,xy_anchors=xy_anchors)
    model.set_mechanical_anchor_properties(anchor_data)

    # Set Loads
    model.anchor_forces = anchor_forces

    # Run Calculations
    print('Performing engineering checks')
    check_anchors(model)

    print('Returning results')
    f, w = plts.anchor_basic(None, model, None)
    plot_file = plts.vtk_save(f,filename='Sample Anchor Image')
    print(plot_file)


    return model


def check_anchors(m: ConcreteAnchors):
    m.check_anchor_spacing()
    m.get_governing_anchor_group()
    m.check_anchor_capacities()

if __name__ == '__main__':
    model = main()
    print('Done')