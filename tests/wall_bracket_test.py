import numpy as np
from anchor_pro.ap_types import WallPositions, FactorMethod
from anchor_pro.elements.wall_bracket import GeometryProps, BracketProps, WallBracketElement  # <-- update if needed


def build_bracket(Bx, By, H, dx, dy, dz):
    """
    Build a trivially checkable bracket:
    - normal = +x so G = I
    - equipment at (0.5, 0, 0), wall at (0, 0, 0) so r = [0.5,0,0]
    - rotations are zero so C reduces to picking [ux, uy, uz]
    """
    xyz_e = np.array([0.0, By/2, H])   # equipment point
    xyz_w = np.array([0.0+dx, By/2+dy, H+dz])   # wall point
    n_hat = np.array([0.0, -1.0, 0.0])   # local n along +x → G = I

    geo = GeometryProps(
        xyz_equipment=xyz_e,
        xyz_wall=xyz_w,
        normal_unit_vector=n_hat,
        supporting_wall=WallPositions.YP,   # not used by evaluation in this test
        releases=(False, True, False, False, False, False)  # release compression in n only
        )

    # Flexibilities picked to make kn = 1000 N/m exactly
    bracket = BracketProps(
        bracket_id="TEST-1000-2000-3000",
        wall_flexibility=1/1000.0,   # 0.001
        brace_flexibility=0.0,
        kn=0.0,                      # filled by __post_init__ (ignored on input)
        kp=2000.0,
        kz=3000.0,
        capacity_to_equipment=120.0,
        bracket_capacity=100.0,
        capacity_to_backing=80.0,    # governing
        shear_capacity=None,
        capacity_method=FactorMethod.lrfd,
    )

    elem = WallBracketElement(geo_props=geo, bracket_props=bracket)
    return elem


def test_wall_bracket_theta_last_handcheck():
    Bx = 2
    By = 2
    H = 10
    bx, by, bz = [0,2,2]  # bracket offsets from unit attachment to wall backing

    elem = build_bracket(Bx,By,H,bx,by,bz)

    # Build u_global with theta last: shape (n_dof, n_theta) = (6, 4)
    # Columns are θ cases:
    # θ0: ux=+0.01  → fn= 10 N
    # θ1: ux=-0.02  → compression, release active → fn= 0 N
    # θ2: uy=+0.01  → fp= 20 N,   Mz= +10 N·m
    # θ3: uz=+0.01  → fz= 30 N,   My= -15 N·m
    n_theta = 4
    u_global = np.zeros((6, n_theta), dtype=float)
    u_global[0, 0] = +0.01  # ux for θ0
    u_global[1, 1] = -0.02  # ux for θ1 (compression)
    u_global[1, 2] = +0.01  # uy for θ2
    u_global[3, 3] = +0.01  # uz for θ3
    # rotations rx, ry, rz remain zero to keep C simple

    # Evaluate
    reactions, results = elem.evaluate_wall_bracket(
        u_global=u_global,
        method=FactorMethod.lrfd
    )

    # ---- Expectations (hand-checkable) ----
    # stiffness
    kn, kp, kz = 1000.0, 2000.0, 3000.0

    # local displacements in npz (G=I and rotations 0 → delta_npz = [ux, uy, uz])
    dn = np.array([+0.01, -0.02, 0.00, 0.1])
    dp = np.array([0.00,  0.00, +0.01, 0.00])
    dz = np.array([0.00,  0.00, 0.00,  0.00])

    # releases: neg-n released → kn=0 where dn<0
    kn_eff = np.where(dn < 0, 0.0, kn)
    kp_eff = np.full_like(dn, kp)
    kz_eff = np.full_like(dn, kz)

    fn_exp = kn_eff * dn  # [10, 0, 0, 0]
    fp_exp = kp_eff * dp  # [0, 0, 20, 0]
    fz_exp = kz_eff * dz  # [0, 0, 0, 0]

    # Global backing forces: with G=I, they match local
    Fx = fn_exp
    Fy = fp_exp
    Fz = fz_exp

    # Moments: r x F, with r=[0.5,0,0]
    r = np.array([-bx, -by, -bz])
    M = np.cross(r, np.vstack([Fx, Fy, Fz]).T).T  # (3, n_theta)
    Mx_exp, My_exp, Mz_exp = M[0], M[1], M[2]

    # Tension unity: governing capacity = 80 N, no ASD scaling in demand (as requested)
    cap = 80.0
    unity_exp = np.clip(fn_exp, 0.0, None) / cap  # [0.125, 0, 0, 0]

    # ---- Assertions ----
    assert results is not None, "Expected results (capacity method matched)."
    np.testing.assert_allclose(results.fn, fn_exp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(results.fp, fp_exp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(results.fz, fz_exp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(results.tension_unity, unity_exp, rtol=0, atol=1e-12)

    # Reactions shape: (6, n_theta)
    assert reactions.reactions_backing.shape == (6, n_theta)
    assert reactions.reactions_equipment.shape == (6, n_theta)

    # Backing reactions
    rb = reactions.reactions_backing
    np.testing.assert_allclose(rb[0, :], Fx, rtol=0, atol=1e-12)   # Fx
    np.testing.assert_allclose(rb[1, :], Fy, rtol=0, atol=1e-12)   # Fy
    np.testing.assert_allclose(rb[2, :], Fz, rtol=0, atol=1e-12)   # Fz
    np.testing.assert_allclose(rb[3, :], Mx_exp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rb[4, :], My_exp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rb[5, :], Mz_exp, rtol=0, atol=1e-12)

    # Equipment reactions: equal and opposite forces, same couple moments
    re = reactions.reactions_equipment
    np.testing.assert_allclose(re[0, :], -Fx, rtol=0, atol=1e-12)
    np.testing.assert_allclose(re[1, :], -Fy, rtol=0, atol=1e-12)
    np.testing.assert_allclose(re[2, :], -Fz, rtol=0, atol=1e-12)
    np.testing.assert_allclose(re[3, :], Mx_exp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(re[4, :], My_exp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(re[5, :], Mz_exp, rtol=0, atol=1e-12)

    print("All wall bracket tests passed with θ last convention and hand-checkable numbers.")


if __name__ == "__main__":
    test_wall_bracket_theta_last_handcheck()