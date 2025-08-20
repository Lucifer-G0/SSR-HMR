import numpy as np
from . import quat

"""
Dual quaternions are represented as arrays of shape [..., 8]
where the last dimension is the dual quaternion representation.
The first 4 elements are the real part and the last 4 elements are the dual part.
[..., [w_r, x_r, y_r, z_r, w_d, x_d, y_d, z_d]]
"""


def from_rotation_translation(rotations: np.array, translations: np.array) -> np.array:
    """
    Convert the rotations (quaternions) and translation (3D vectors) information to dual quaternions.

    Parameters
    ----------
    rotations: np.array[..., [w, x, y, z]]]
    translations : np.array[..., 3]

    Returns
    -------
    dq : np.array[..., 8]
    """
    # dual quaternion (sigma) = qr + qd (+ is not an addition, but concatenation)
    # real part of dual quaternions represent rotations
    # and is represented as a conventional unit quaternion
    q_r = rotations
    # dual part of dual quaternions represent translations
    # t is a pure quaternion (0, x, y, z)
    # q_d = 0.5 * eps * t * q_r
    t = np.zeros((translations.shape[:-1] + (4,)))
    t[..., 1:] = translations
    q_d = 0.5 * quat.mul(t, q_r)
    dq = np.concatenate((q_r, q_d), axis=-1)
    return dq


def from_translation(translations: np.array) -> np.array:
    """
    Convert a translation to a dual quaternion.

    Parameters
    ----------
    translations : np.array[..., 3]

    Returns
    -------
    dual_quats : np.array[..., 8]
    """
    dual_quats = np.zeros((translations.shape[:-1] + (8,)))
    # real part of dual quaternions represent rotations
    # and is represented as a conventional unit quaternion
    dual_quats[..., 0:1] = 1
    # dual part of dual quaternions represent translations
    # t is a pure quaternion (0, x, y, z)
    # q_d = 0.5 * eps * t * q_r (q_r = 1, thus, q_d = 0.5 * eps * t)
    dual_quats[..., 5:] = translations * 0.5
    return dual_quats


def to_rotation_translation(dq: np.array) -> np.array:
    """
    Convert a dual quaternion to the rotations (quaternions) and translations (3D vectors).

    Parameters
    ----------
    dq: np.array[..., 8]

    Returns
    -------
    rotations: np.array[..., [w, x, y, z]]]
    translations: np.array[..., 3]
    """
    dq = dq.copy()
    q_r = dq[..., :4]
    # rotations can ge get directly from the real part of the dual quaternion
    rotations = q_r
    q_d = dq[..., 4:]
    # the translation (pure quaternion) t = 2 * q_d * q_r*
    # where q_r* is the conjugate of q_r
    translations = (2 * quat.mul(q_d, quat.conjugate(q_r)))[..., 1:]
    return rotations, translations


def normalize(dq: np.array) -> np.array:
    """
    Normalize the dual quaternion to unit length and make sure that
    the dual part is orthogonal to the real part (unit dual quaternion).

    Parameters
    ----------
    dq: np.array[..., 8]

    Returns
    -------
    dq: np.array[..., 8]
    """
    dq = dq.copy()
    q_r = dq[..., :4]
    q_d = dq[..., 4:]
    norm = np.linalg.norm(q_r, axis=-1)
    qnorm = np.stack((norm, norm, norm, norm), axis=-1)
    q_r_normalized = q_r / qnorm
    q_d_normalized = q_d / qnorm
    if not is_unit(np.concatenate((q_r_normalized, q_d_normalized), axis=-1)):
        # make sure that the dual quaternion is orthogonal to the real quaternion
        dot_q_r_q_d = np.sum(q_r * q_d, axis=-1)  # dot product of q_r and q_d
        q_d_normalized_ortho = q_d_normalized - (
            q_r_normalized * (dot_q_r_q_d / (norm * norm))[..., np.newaxis]
        )
        dq = np.concatenate((q_r_normalized, q_d_normalized_ortho), axis=-1)
    else:
        dq = np.concatenate((q_r_normalized, q_d_normalized), axis=-1)
    return dq


def is_unit(dq: np.array, atol: float = 1e-03) -> bool:
    """
    Check if the dual quaternion is a unit one.
    A unit dual quaternion satisfies two properties:
    - The norm of the real part is 1
    - The dot product of the real and dual part is 0.

    Parameters
    ----------
    dq: np.array[..., 8]
    """
    q_r = dq[..., :4]
    q_d = dq[..., 4:]
    sqr_norm_q_r = np.sum(q_r * q_r, axis=-1)
    if np.isclose(sqr_norm_q_r, 0).all():
        return True
    rot_normalized = np.isclose(sqr_norm_q_r, 1).all()
    trans_normalized = np.isclose(np.sum(q_r * q_d, axis=-1), 0, atol=atol).all()
    return rot_normalized and trans_normalized

def unroll(dq: np.array, axis: int) -> np.array:
    """
    通过选择最小距离（或等效地，最大点积）的表示（dq 或 -dq），
    在时间维度上强制执行双四元数的连续性。

    参数
    ----------
    dq : np.array[..., 8]
        输入的双四元数数组，其形状应为 (..., 8)，其中 8 表示双四元数的 8 个分量。
    axis : int
        需要展平的轴（例如，帧轴）。

    返回
    -------
    dq : np.array[..., 8]
        展平后的双四元数数组。
    """
    # 交换第一个轴和指定轴的位置，以便后续处理
    dq = dq.swapaxes(0, axis)
    q_r = dq[..., :4]
    # 从第二个四元数开始处理，因为第一个四元数的覆盖是保留的
    for i in range(1, len(q_r)):
        # 计算前一个和当前四元数之间的距离（点积）
        d0 = np.sum(q_r[i] * q_r[i - 1], axis=-1)
        # 计算前一个和当前四元数取反后的距离（点积）
        d1 = np.sum(-q_r[i] * q_r[i - 1], axis=-1)
        # 如果取反后的四元数的距离更小，则使用它
        dq[i][d0 < d1] = -dq[i][d0 < d1]
    # 将轴交换回原始位置
    dq = dq.swapaxes(0, axis)
    # 返回处理后的双四元数数组
    return dq
