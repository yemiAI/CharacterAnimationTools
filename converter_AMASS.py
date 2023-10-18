from anim import amass
from anim import bvh
from anim.animation import Animation
anim_bvh: Animation = bvh.load(filepath="data/motorica_locking/kthstreet_gLO_sFM_cAll_d01_mLO_ch01_dennisscorpiocoffey_001.bvh")

amass.save_as_npz(anim_bvh,"kthstreet_gLO_sFM_cAll_d01_mLO_ch01_dennisscorpiocoffey_001.npz",16,"female", 120)