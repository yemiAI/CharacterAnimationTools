"""
amass.py
AMASS :
    Format: ***.npz (numpy binary file)
    Parameters:
        "trans" (np.ndarray): Root translations. shape: (num_frames, 3).
        "gender" (np.ndarray): Gender name. "male", "female", "neutral".
        "mocap_framerate" (np.ndarray): Fps of mocap data.
        "betas" (np.ndarray): PCA based body shape parameters. shape: (num_betas=16).
        "dmpls" (np.ndarray): Dynamic body parameters of DMPL. We do not use. Unused for skeleton. shape: (num_frames, num_dmpls=8).
        "poses" (np.ndarray): SMPLH pose parameters (rotations). shape: (num_frames, num_J * 3 = 156).
"""
# loading amass data.

from __future__ import annotations

from pathlib import Path
import numpy as np
from anim.skel import Skel
from anim.animation import Animation
from anim.smpl import load_model, calc_skel_offsets, SMPL_JOINT_NAMES, SMPLH_JOINT_NAMES
from util import quat




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
    """
    args:
        amass_motion_file (Path): Path to the AMASS motion file.
        smplh_path (Path): Optional. Path to the SMPLH model.
        remove_betas (bool): remove beta parameters from AMASS.
        gender (str): Gender of SMPLH model.
        num_betas (int): Number of betas to use. Defaults to 16.
        num_dmpls (int): Number of dmpl parameters to use. Defaults to 8.
        scale (float): Scaling paramter of the skeleton. Defaults to 100.0.
        load_hand (bool): Whether to use hand joints. Defaults to True.
        load_dmpl (bool): Whether to use DMPL. Defaults to False.
    Return:
        anim (Animation)
    """
    if load_hand: 
        NUM_JOINTS = 52
        names = SMPLH_JOINT_NAMES[:NUM_JOINTS]
    else: 
        NUM_JOINTS = 24
        names = SMPL_JOINT_NAMES[:NUM_JOINTS]

    if not isinstance(amass_motion_path, Path):
        amass_motion_path = Path(amass_motion_path)
    if not isinstance(smplh_path, Path):
        smplh_path = Path(smplh_path)
    
    
    trans_quat = quat.mul(quat.from_angle_axis(-np.pi / 2, [0, 1, 0]), quat.from_angle_axis(-np.pi / 2, [1, 0, 0]))

    # Load AMASS info.
    amass_dict = np.load(amass_motion_path, allow_pickle=True)
    if gender == None: gender = str(amass_dict["gender"])
    fps = int(amass_dict["mocap_frame_rate"])
    if remove_betas: betas = np.zeros([num_betas])
    else: betas = amass_dict["betas"][:num_betas]
    axangles = amass_dict["poses"]#
    num_frames = len(axangles)
    axangles = axangles.reshape([num_frames, -1, 3])[:,:NUM_JOINTS]#
    print(axangles)
    print(axangles.shape)
    quats = quat.from_axis_angle(axangles) #R
    root_rot = quat.mul(trans_quat[None], quats[:,0])
    quats[:,0] = root_rot

    if skel == None:
        # Load SMPL parmeters.
        smplh_dict = load_model(smplh_path, gender)
        parents = smplh_dict["kintree_table"][0][:NUM_JOINTS]
        parents[0] = -1
        J_regressor = smplh_dict["J_regressor"]
        shapedirs = smplh_dict["shapedirs"]
        v_template = smplh_dict["v_template"]
        
        J_positions = calc_skel_offsets(betas, J_regressor, shapedirs, v_template)[:NUM_JOINTS] * scale
        root_offset = J_positions[0]
        offsets = J_positions - J_positions[parents]
        offsets[0] = root_offset
        skel = Skel.from_names_parents_offsets(names, parents, offsets, skel_name="SMPLH")
    
    root_pos = skel.offsets[0][None].repeat(len(quats), axis=0)
    trans = amass_dict["trans"] * scale + root_pos
    trans = trans @ quat.to_xform(trans_quat).T  # Reverse
    anim = Animation(skel=skel, quats=quats, trans=trans, fps=fps, anim_name=amass_motion_path.stem)
    
    save_as_npz(anim, "output_path_here.npz", betas, gender, fps)  # Please replace with your desired output path
    

    return anim



bone_mapAMASS =[0, 1, 4, 7, None, 2,5,8,None,3,6,9,12,15,13,16,18,20,None,None,None,25,26,27,28,29,30,31,32,33,34,35,36,14,17,19,21,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51] 
    
#bone_mapMotorica = [0,43,44,45,46,47,48,49,50,1,None,2,3,4,5,6,7,8,12,13,14,15,16,17,21,22,23,18,19,21,9,10,11,24,25,26,27,31,32,33,34,35,36,40,41,42,37,38,39,28,29,30] 

#bone_mapMotorica = [0, 9, 11, 12, 13, 14, 15, 16, 17, 30, 31, 32, 18, 19, 20, 21, 22, 23, 27, 28, None, 29, 25, 26, 33, 34, 35, 36, 49, 50, 51, 37, 38, 39, 40, 41, 42, 46, 47, 48, 43, 44, 45, 1, 2, 3, 4, 5, 6, 7, 8]

#bone_mapMotorica = [0, 

bone_mapMotorica = [0, 9, 11, 12, 13, 14, 15, 16, 17, 30, 31, 32, 18, 19, 20, 21, 22, 23, 27, 28, None, 29, 25, 26, 33, 34, 35, 36, 49, 50, 51, 37, 38, 39, 40, 41, 42, 46, 47, 48, 43, 44, 45, 1, 2, 3, 4, 5, 6, 7, 8]

#bone_mapMotorica = [None] * 51
#bone_mapMotorica[8] = 6 

def bone_innermap(mapping, output_poses, input_poses):
    for b_from, b_to in enumerate(mapping):
            # Adjust the mapping target index.
            #b_to = b_toX - 1

            # Check if both source and target indices are within the valid range.
            #if 0 <= b_from < num_bones_in_input and 0 <= b_to < num_bones_in_output:
                # If they are, copy the corresponding pose values.
        if b_to is None :
            pass                
        else:
            output_poses[:,3 * b_to: 3 * b_to + 3] = input_poses[:,3 * b_from: 3 * b_from + 3]  


def bone_mapping(input_poses) :


    # Make sure the bone_map indices are within proper range.
    #num_bones_in_input = (input_poses.shape[1]) // 3
    #num_bones_in_output =(out_poses_a.shape[1]) // 3

    if input_poses.shape[1] == 153:    
        print("Converting Motorica format bvh")
        out_poses_a = np.zeros([input_poses.shape[0],165])   #rearranging input poses
        bone_innermap(bone_mapMotorica, out_poses_a, input_poses)
        out_poses = np.zeros([input_poses.shape[0],165])   #rearranging input poses
        bone_innermap(bone_mapAMASS, out_poses, out_poses_a)
    elif(input_poses.shape[1] == 156):
        print("Converting AMASS format bvh")
        out_poses = np.zeros([input_poses.shape[0],165])   #rearranging input poses
        bone_innermap(bone_mapAMASS, out_poses, input_poses)
    else:
        print("Can't work out input format, quitting")
        exit(0)
            
    return out_poses  

def save_as_npz(anim: Animation, save_path: str,num_betas: int, gender: str, mocap_frame_rate=None,scale: float=100, static= False):
    """
    Save the animation data to a .npz file format.

    Args:
    anim (Animation): The Animation object containing the motion data.
    save_path (str): The path where the .npz file will be saved.
    """
    # Check if the save path is a string, and convert it to a Path object if necessary.
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    # Rev
   
    #trans_quat = quat.mul(quat.from_angle_axis(-np.pi / 2, [0, 1, 0]), quat.from_angle_axis(-np.pi / 2, [1, 0, 0]))
    
    #fps = int(amass_dict["mocap_frame_rate"])
    
    reversed_trans_quat = quat.mul(quat.from_angle_axis(np.pi / 2, [1, 0, 0]), quat.from_angle_axis(np.pi / 2, [0, 1, 0]))
   

    #trans = amass_dict["trans"] * scale + root_pos


    #trans_quat = quat.mul(quat.from_angle_axis(-np.pi / 2, [0, 1, 0]), quat.from_angle_axis(-np.pi / 2, [1, 0, 0]))
    
    #inverse_quat = quat.conjugate(reversed_trans_quat)

    
    #trans = trans @ quat.to_xform(trans_quat).T  # Reverse this operation in a single line
    #amass_diction = np.load(amass_motion_path, allow_pickle=True)
    
    #main_trans = amass_diction["trans"] * scale + root_pos
    
    #trans = main_trans @ np.linalg.inv(quat.to_xform(reversed_trans_quat))

    #amass_diction = np.load(amass_motion_path, allow_pickle=True)
    
    #main_axis_angles = amass_diction["poses"]#
    
    root_poses = quat.to_axis_angle(anim.quats[:, 0, :] * reversed_trans_quat)
    
    axis_angles_conv = quat.to_axis_angle(anim.quats)
   
    axis_angles_conv[:,0,:] = root_poses[:,:]
   
   
    NUM_JOINTS = len(anim.joint_names)
    
    num_frames= anim.quats.shape[0]
    
    axis_angles = axis_angles_conv.reshape(num_frames, NUM_JOINTS*3)  # reshape to have individual joints' data.
    
    #axis_angles = axangles.reshape([num_frames, -1, 3])[:,:NUM_JOINTS]#
    
    print(axis_angles)
    print(axis_angles.shape)
    
    
    #write this bit propoerly (inspect script to be sure files are writen in correctly)! write this rtoday 
    
    
    poses = np.concatenate([axis_angles, anim.hand_poses], axis=0) if hasattr(anim, 'hand_poses') else axis_angles
    
    
    out_poses = bone_mapping(poses)
    
    #out_pose_body = out_poses[:, 3:66]
    #out_pose_hand = out_poses[:, 75:165]
    #out_pose_jaw = np.zeros([out_poses.shape[0],6])
    #out_pose_eyes = np.zeros([out_poses.shape[0],6])




    trans = anim.trans @ quat.to_xform(reversed_trans_quat).T  # Reverse

    root_pos = np.zeros([out_poses.shape[0],3]) #workout what the 
    
    
    
    
    
    
    if static: 
        out_trans= np.zeros_like(trans)
    else:
        out_trans = (trans - root_pos) / scale
    
    
    amass_template= np.load('data/salsa_1_stageii.npz', allow_pickle=True)
    
    out_root_orient = out_poses[:, 0:3]
    #markers_latent=amass_template['markers_latent']
    latent_labels=amass_template['latent_labels']
    labels=amass_template['labels']
    labels_obs=amass_template['labels_obs']
    markers_latent_vids=amass_template['markers_latent_vids']
    markers_latent=amass_template['markers_latent']
    surface_model_type = amass_template['surface_model_type']
    #num_betas = amass_template['num_betas']
    num_betas= np.array(num_betas) 
    out_mocaptime= np.array(out_poses.shape[0] / anim.fps)
    out_marker_labels = np.array([])
    out_marker_data = np.array([])
    surface_model_type = amass_template['surface_model_type']
    num = amass_template['surface_model_type']
    
    betas = amass_template['betas']
    
    
    markers = amass_template['markers_obs']
    markers_obs = amass_template['markers_obs']
    markers_sim = amass_template['markers_sim']
    marker_meta = amass_template['marker_meta']
    
    if mocap_frame_rate is None:
        mocap_frame_rate = anim.fps
    
    # Ensure betas is in the correct shape, assuming it's not already
    #if betas.ndim == 1:
     #   betas = betas.reshape((1, -1))  # Reshape betas to have shape (1, num_betas)

    
    # Prepare the data for saving.
    data = {
        "trans": out_trans,  # Root translations (shape: [num_frames, 3])
        "gender": np.array(gender),  # Gender (shape: [])
        "mocap_frame_rate": np.array(mocap_frame_rate),  # Framerate (shape: [])
        #"betas": np.ndarray(betas),  # Body shape coefficients (shape: [num_betas=16])
        #"poses": bone_mapping(poses),  # Poses in axis-angle format (shape: [num_frames, num_pose_parameters])
        # "dmpls": dmpls,  # Optional: Dynamic expression coefficients for faces or muscles (shape: [num_frames, num_dmpls=8])
    }

    # Save the data to a .npz file.
    #np.savez(save_path, **data)
    # np.savez(save_path , gender =data['gender'], mocap_framerate = data['mocap_frame_rate'], betas=betas,marker_labels = out_marker_labels, marker_data = out_marker_data, poses = out_poses, trans = data['trans'], mocap_time_length = out_mocaptime, root_orient=out_root_orient, pose_body=out_pose_body, pose_hand= out_pose_hand, pose_eyes= out_pose_eyes, latent_label=latent_labels, markers=markers, markers_obs=markers_obs, markers_sim=markers_sim, marker_meta= marker_meta,  surface_model_type= surface_model_type,labels=labels, markers_latent_vids=markers_latent_vids, markers_latent =markers_latent, num_betas=num_betas)
    
    np.savez(save_path , gender =data['gender'], mocap_framerate = data['mocap_frame_rate'], betas=betas,marker_labels = out_marker_labels, marker_data = out_marker_data, poses = out_poses, trans = data['trans'], mocap_time_length = out_mocaptime, root_orient=out_root_orient,latent_label=latent_labels, markers=markers, markers_obs=markers_obs, markers_sim=markers_sim, marker_meta= marker_meta,  surface_model_type= surface_model_type,labels=labels, markers_latent_vids=markers_latent_vids, markers_latent =markers_latent, num_betas=num_betas)

    print(f"Data saved at {save_path}")