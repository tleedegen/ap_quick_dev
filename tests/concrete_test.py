import numpy as np
import xlwings as xw
import pandas as pd
from anchor_pro.elements.concrete_anchors import (
    GeoProps,
    MechanicalAnchorProps,
    ConcreteProps,
    ConcreteAnchors,
    Profiles,
    AnchorBasicInfo,
    Phi,
    AnchorTypes,
    InstallationMethod)

# Create Geometry Element
sx = 3
sy = 3
xy_anchors = np.array([[-sx/2, sy/2],
                       [sx/2, sy/2],
                       [-sx/2, -sy/2],
                       [sx/2, -sy/2]])
Bx = 16
By = 18
cx_pos = 1.5
cx_neg = np.inf
cy_pos = np.inf
cy_neg = np.inf

geo_props = GeoProps(
    xy_anchors=xy_anchors,
    Bx=Bx,
    By=By,
    cx_pos=cx_pos,
    cx_neg=cx_neg,
    cy_neg=cy_neg,
    cy_pos=cy_pos,
    )

concrete_props = ConcreteProps(
    weight_classification='NWC',
    profile=Profiles.slab,
    fc=4000,
    lw_factor=1.0,
    cracked_concrete=True,
    poisson=0.25,
    t_slab=6
)

'''Data from the anchor catalog'''
xlpath = r"C:\Users\djmiller\OneDrive - Degenkolb Engineers\AnchorPro\Reference Files\Mechanical Anchor Catalog.xlsx"
wb = xw.Book(xlpath)
sheet = wb.sheets['Anchors']
table_cell = sheet.range('tbl_anchors')
start_address = table_cell.get_address(0, 0, include_sheetname=True, external=False)
df = sheet.range(start_address).expand().options(pd.DataFrame,
                                                 header=1,
                                                 index=False,
                                                 expand='table').value

# Create Anchor Properties Element
anchor_id = "HILTI KBTZ2-C [0.5X2.5]"
data = df[df['anchor_id']==anchor_id].iloc[0]

info = AnchorBasicInfo(
    anchor_id=anchor_id,
    installation_method=data['installation_method'],
    manufacturer=data['manufacturer'],
    product=data['product'],
    product_type="",
    anchor_type=data['anchor_type'],
    esr=data['esr'],
    cost_rank=data['cost_rank']
)

phi = Phi(
    saN=data['phi_saN'],
    pN=data['phi_pN'],
    cN=data['phi_cN'],
    cV=data['phi_cV'],
    saV=data['phi_saV'],
    cpV=data['phi_cpV'],
    seismic=0.75,
    eqV = data['phi_eqV'],
    eqN = data['phi_eqN']
)
# phi = Phi(
#     saN=data['phi_saN'],
#     pN=0.65,
#     cN=0.65,
#     cV=0.7,
#     saV=0.65,
#     cpV=0.7,
#     seismic=0.75,
#     eqV = 0.75,
#     eqN = 0.75
# )

#todo: see below
''' when building the coordinator funtion to populate this from the input table,
be sure to incorporate logic such as selection of Vsa from Vsa_seismic and Vsa_default.
Similar for other parameters like which c1_min values to use, etc. 
Do all he interpolated, and selection outside of the concrete object'''



anchor_props = MechanicalAnchorProps(
    info=info,
    fya=data['fya'],
    fua = data['fua'],
    Nsa = data['Nsa'],
    Np = data['Np_eq'],  # Needs special handling
    kc = data['kc_cr'], # Needs special handling (use uncr or cr)
    kc_uncr = data['kc_uncr'],
    kc_cr = data['kc_cr'],
    le = data['le'],
    da = data['da'],
    cac = data['cac1_slab'],  # Needs special handling
    esr = data['esr'],
    hef_default = data['hef_default'],
    Vsa = data['Vsa_eq'], # Needs special handling
    K = data['K_cr'], # Needs special handling
    K_cr = data['K_cr'],
    K_uncr = data['K_uncr'],
    Kv = data['Kv'],
    hmin = data['hmin1_slab'], # Needs special handling
    c1 = data['c11_slab'],  # Needs special handling
    s1 = data['s11_slab'],  # Needs special handling
    c2 = data['c21_slab'],  # Needs special handling
    s2 = data['s21_slab'],  # Needs special handling
    phi = phi,
    abrg = data['abrg'])


# Create Concrete Anchor Element
anchor_obj = ConcreteAnchors(geo_props=geo_props,
                             concrete_props=concrete_props)

anchor_obj.set_anchor_props(anchor_props)


# Define input forces (n_anchor,3,n_forces)
forces = np.tile([[1000], [500], [0]], (4, 1, 1))

# Run Checks
results = anchor_obj.evaluate(forces)