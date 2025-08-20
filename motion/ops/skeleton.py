import numpy as np
import pymotion.rotations.quat as quat
import pymotion.rotations.dual_quat as dquat

"""
A skeleton is a set of joints connected by bones.
The skeleton is defined by:
    - the local offsets of the joints
    - the parents of the joints
    - the local rotations of the joints
    - the global position of the root joint

This functions convert skeletal information to 
root-centered dual quaternions and vice-versa.

Root-centered dual quaternions are useful when training
neural networks, as all information is local to the root
and the neural network does not need to learn the FK function.

"""


def from_root_dual_quat(dq: np.array, parents: np.array) -> np.array:
    """
    Convert root-centered dual quaternion to the skeleton information.

    Parameters
    ----------
    dq: np.array[..., n_joints, 8]
        Includes as first element the global position of the root joint
    parents: np.array[n_joints]

    Returns
    -------
    rotations : np.array[..., n_joints, 4]
    translations : np.array[..., n_joints, 3]
    """
    n_joints = dq.shape[1]
    # rotations has shape (frames, n_joints, 4)
    # translations has shape (frames, n_joints, 3)
    rotations, translations = dquat.to_rotation_translation(dq.copy())
    # make transformations local to the parents
    # (initially local to the root)
    for j in reversed(range(1, n_joints)):
        parent = parents[j]
        if parent == 0:  # already in root space
            continue
        inv = quat.inverse(rotations[..., parent, :])
        translations[..., j, :] = quat.mul_vec(
            inv,
            translations[..., j, :] - translations[..., parent, :],
        )
        rotations[..., j, :] = quat.mul(inv, rotations[..., j, :])
    return translations, rotations


def to_root_dual_quat(
        rotations: np.array, global_pos: np.array, parents: np.array, offsets: np.array
):
    """
    将旋转、骨架信息（偏移），根据父子关系转换为以根节点为中心的双四元数表示。

    参数：
        rotations : np.array[..., n_joints, 4] 关节的旋转。
        global_pos: np.array[..., 3] 根节点的全局位置。
        parents : np.array[n_joints] 关节的父节点。
        offsets : np.array[n_joints, 3] 关节相对于其父节点的偏移。

    返回：
        dual_quat : np.array[..., n_joints, 8]  如(n_frames,n_joints,8)
        骨架的以根节点为中心的双四元数表示。
    """
    # 断言确保根节点的偏移为零向量
    assert (offsets[0] == np.zeros(3)).all()
    # 计算关节数量
    n_joints = rotations.shape[1]

    # 复制旋转数据以避免修改原始输入
    rotations = rotations.copy()
    # 将偏移量扩展到与旋转数据相同的批次维度
    translations = np.tile(offsets, rotations.shape[:-2] + (1, 1))
    # 将全局位置赋值给根节点的平移向量
    translations[..., 0, :] = global_pos

    # 将所有变换转换为相对于根节点的局部变换
    for j in range(1, n_joints):
        parent = parents[j]
        if parent == 0:  # 如果已经是根空间，则跳过
            continue
        # 计算当前关节相对于其父节点的局部平移
        translations[..., j, :] = (quat.mul_vec(rotations[..., parent, :], translations[..., j, :])+ translations[..., parent, :])
        # 计算当前关节相对于其父节点的局部旋转
        rotations[..., j, :] = quat.mul(rotations[..., parent, :], rotations[..., j, :])

    # 将旋转和平移转换为双四元数
    dual_quat = dquat.from_rotation_translation(rotations, translations)
    # 返回计算得到的双四元数
    return dual_quat