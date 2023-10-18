import numpy as np
from pathlib import Path
from anim.skel import Skel
from anim.animation import Animation

def load(
    amass_motion_path: Path,
    skel: Skel=None,
    smplh_path: Path=Path("data/smplh/neutral/model.npz"),
    remove_betas: bool=False,
    gender: str=None,
    num_betas: int=16,
    scale: float=100.0,
    load_hand=True,
) -> Animation:

    amass_dict = np.load(amass_motion_path, allow_pickle=True)


    if gender == None: gender = str(amass_dict["gender"])
    fps = int(amass_dict["mocap_frame_rate"])


    if remove_betas: betas = np.zeros([num_betas])
    else: betas = amass_dict["betas"][:num_betas]
    axangles = amass_dict["poses"]

    num_frames = len(axangles)

    axangles = axangles.reshape([num_frames, -1, 3])[:,:NUM_JOINTS]

    print(axangles)
    quats = quat.from_axis_angle(axangles)
    print(quats)
    root_rot = quat.mul(trans_quat[None], quats[:,0])
    print(root_rot)
    quats[:,0] = root_rot

    #if skel == None:
        # Load SMPL parmeters.
        #smplh_dict = load_model(smplh_path, gender)
        #parents = smplh_dict["kintree_table"][0][:NUM_JOINTS]
        #parents[0] = -1
        #J_regressor = smplh_dict["J_regressor"]
        #shapedirs = smplh_dict["shapedirs"]
        #v_template = smplh_dict["v_template"]

        #J_positions = calc_skel_offsets(betas, J_regressor, shapedirs, v_template)[:NUM_JOINTS] * scale
        #root_offset = J_positions[0]
        #offsets = J_positions - J_positions[parents]
        #offsets[0] = root_offset
        #skel = Skel.from_names_parents_offsets(names, parents, offsets, skel_name="SMPLH")

            #root_pos = skel.offsets[0][None].repeat(len(quats), axis=0)
    trans = amass_dict["trans"] * scale + root_pos
    trans = trans @ quat.to_xform(trans_quat).T
    anim = Animation(skel=skel, quats=quats, trans=trans, fps=fps, anim_name=amass_motion_path.stem)

#return anim