# =====================================================================
# 6. APPLICATION DASHBOARD WORKSPACE (SINGLE PAGE FLOW)
# =====================================================================

st.set_page_config(page_title="SanjeevaniDock Hub", layout="wide")
st.title("🌿 SanjeevaniDock - Rare Disease Phytochemical Docking Platform")
st.markdown("**SanjeevaniDock bridges traditional Ayurvedic pharmacology with modern structure-based drug discovery pipelines targeting rare and orphan diseases.**")
st.markdown("**Developed by: Ms. Amisha Bilakar & Team**")

# Master Reset
if st.button("🔄 Reset Entire Environment", type="secondary", use_container_width=True):
    for key in list(st.session_state.keys()): del st.session_state[key]
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt", "temp_lig_state.pdb", "redesign_ligand.pdbqt", "redesign_docking_poses.pdbqt"]:
        if os.path.exists(f): os.remove(f)
    st.success("Dashboard cache and runtime structures completely cleared!")
    safe_rerun()
