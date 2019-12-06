import numpy as np
import random


def simplex_projection(vector, s):
    size = len(vector)
    all_positions = True
    # check if in simplex
    for i in range(0, size):
        if vector[i] < 0:
            all_positions = False
            break
    if np.sum(vector) <= s and all_positions:
        return vector
    # compute lagrange multipliers
    theta = compute_lagrange(vector, s)
    # do simplex projection
    result = np.zeros(size)
    for i in range(0, size):
        if (vector[i] - theta) <= 0:
            result[i] = 0
        else:
            result[i] = vector[i] - theta
    return result


def l1_projection(vector, s):
    size = len(vector)
    one_norm = np.linalg.norm(vector, 1)
    if one_norm <= s:
        return vector
    simplex_proj = simplex_projection(np.absolute(vector), s)
    
    for i in range(0, size):
        if vector[i] > 0:
            sign = 1
        else:
            sign = -1
        simplex_proj[i] = simplex_proj[i] * sign
    return simplex_proj


def l0_projection(vector, s):
    size = vector.shape[0]
    if np.count_nonzero(vector) <= s:
        return vector
    # else find s_largest element
    abs_vec = np.abs(vector)
    abs_vec = np.partition(abs_vec, vector.shape[0] - s)
    s_largest = abs_vec[vector.shape[0] - s]
    result = np.zeros(size)
    select_count = 0
    for i in range(0, size):
        if abs(vector[i]) >= s_largest:
            result[i] = vector[i]
            select_count += 1
        if select_count == s:
            break
    return result


def smooth_hinge_loss(sparse_matrix, vector_y, vector_x):
    size = np.shape(sparse_matrix)[0]
    # sparse matrix times x
    vector_w = sparse_matrix.dot(vector_x.reshape((vector_x.shape[0], 1)))
    vector_wy = np.zeros(size)
    # component wise product
    for i in range(0, size):
        vector_wy[i] = vector_w[i] * vector_y[i]
    loss = 0
    for i in range(0, size):
        if vector_wy[i] <= 0:
            loss += 0.5 - vector_wy[i]
        elif vector_wy[i] <= 1:
            loss += 0.5 * (1 - vector_wy[i]) * (1 - vector_wy[i])
    loss /= size
    return loss


def square_norm(vector):
    return np.sum(np.square(vector))


def smooth_hinge_loss_reg(sparse_matrix, vector_y, vector_x, mu):
    loss = smooth_hinge_loss(sparse_matrix, vector_y, vector_x)
    reg_loss = 0.5 * mu * square_norm(vector_x)
    return loss + reg_loss


def prediction_accuracy(sparse_matrix, vector_y, vector_x):
    size = np.shape(sparse_matrix)[0]
    num_correct = 0
    vector_w = sparse_matrix.dot(vector_x)
    vector_w = vector_w.reshape(vector_w.shape[0])
    for i in range(0, size):
        if vector_w[i] * vector_y[i] > 0:
            num_correct += 1
    return num_correct / size


def primal_grad_smooth_hinge_loss_reg(sparse_matrix, spares_matrix_transpose, vector_y, vector_x, mu):
    size = np.shape(sparse_matrix)[0]
    vector_w = sparse_matrix.dot(vector_x.reshape((vector_x.shape[0], 1)))
    vector_w = vector_w.reshape(vector_w.shape[0])

    vector_wy = np.multiply(vector_w, vector_y)
    gradient = np.zeros(size)
    for i in range(0, size):
        if vector_wy[i] <= 0:
            gradient[i] = -vector_y[i]
        elif vector_wy[i] <= 1:
            gradient[i] = (vector_wy[i] - 1) * vector_y[i]
        else:
            gradient[i] = 0
    temp = spares_matrix_transpose.dot(gradient.reshape((gradient.shape[0], 1)))
    temp = temp.reshape(temp.shape[0])
    temp2 = mu*vector_x
    grad = np.add(temp / size, temp2)
    return grad


def primal_grad_smooth_hinge_loss_reg_k(sparse_matrix, vector_y, vector_x, mu, k, indices):
    samples = np.shape(sparse_matrix)[1]
    vector_w = np.zeros(k)
    """
    for i in range(0, k):
        non_zero = np.nonzero(sparse_matrix[i])
        for j in non_zero[0]:
            vector_w[i] += sparse_matrix[i][j] * vector_x[j]
    """
    vector_w = sparse_matrix[indices, :].dot(vector_x.reshape((vector_x.shape[0],1)))
    vector_wy = np.zeros(k)
    for i in range(0, k):
        vector_wy[i] = vector_w[i] * vector_y[indices[i]]

    gradient = np.zeros(k)
    for i in range(0, k):
        if vector_wy[i] <= 0:
            gradient[i] = -vector_y[indices[i]]
        elif vector_wy[i] <= 1:
            gradient[i] = (vector_wy[i] - 1) * vector_y[indices[i]]
        else:
            gradient[i] = 0
    result = sparse_matrix[indices[0:k], :].transpose().dot(gradient.reshape((gradient.shape[0],1)))
    result = result.reshape((result.shape[0]))
    return result / k + mu * vector_x


def simplex_projection_theta(vector, s):
    size = len(vector)
    all_positions = True
    for i in range(0, size):
        if vector[i] < 0:
            all_positions = False
            break

    total = np.sum(vector)
    if total <= s and all_positions:
        return 0

    return compute_lagrange(vector, s)


def l1_l0_projection(vector, s, tau):
    size = len(vector)
    if np.count_nonzero(vector) <= s:
        return l1_projection(vector, s)

    abs_vec = np.abs(vector)
    abs_vec = np.partition(abs_vec, vector.shape[0] - s)
    s_largest = abs_vec[vector.shape[0] - s]

    compact_l0_abs = np.zeros(s)
    compact_l0_idx = np.zeros(s)
    l0 = np.zeros(size)
    num_selected = 0
    for i in range(0, size):
        curr = vector[i]
        if abs(curr) >= s_largest:
            l0[i] = curr
            compact_l0_abs[num_selected] = abs(curr)
            compact_l0_idx[num_selected] = i
            num_selected += 1
        if num_selected == s:
            break
    if np.linalg.norm(l0, 1) <= tau:
        return l0
    theta = simplex_projection_theta(compact_l0_abs, tau)
    result = np.zeros(size)
    for i in range(0, s):
        idx = compact_l0_idx[i]
        if compact_l0_abs[i] - theta <= 0:
            # og code uses result(idx) instead of [idx] not sure what it does
            result[int(idx)] = 0
        else:
            # og code uses result(idx) instead of [idx] not sure what it does
            result[(int(idx))] = compact_l0_abs[i] - theta
    for i in range(0, s):
        idx = compact_l0_idx[i]
        if l0[int(idx)] > 0:
            sign = 1
        else:
            sign = -1
        # og code uses result(idx) instead of [idx] not sure what it does
        result[int(idx)] *= sign
    return result


def normal_matrix(sparse_matrix):
    cols = np.shape(sparse_matrix)[1]
    ones = np.zeros(cols)
    for i in range(0, cols):
        ones[i] = 1
    row_norm = np.matmul(np.multiply(sparse_matrix, sparse_matrix), ones)
    row_norm = np.sqrt(row_norm)
    return np.matmul(np.linalg.inv(np.diag(row_norm)), sparse_matrix)


def sparse_matrix_generator(sparse_matrix):
    rows = np.shape(sparse_matrix)[0]
    cols = np.shape(sparse_matrix)[1]
    row = [int]
    column = [int]
    value = [float]
    for i in range(0, rows):
        any_selected = False
        for j in range(0, cols):
            v_ij = random.random() % 100
            if v_ij < 20:
                row.append(i)
                column.append(j)
                value.append(v_ij)
                any_selected = True
        if not any_selected:
            j = random.random() % cols
            v_ij = random.random() % 20
            row.append(i)
            column.append(j)
            value.append(v_ij)
    matrix = np.zeros(rows, cols)
    for i in range(0, len(row)):
        matrix[row[i]][column[i]] = value[i]
    return normal_matrix(matrix)


def compute_lagrange(vector, s):
    size = len(vector)
    # copy and sort in decreasing order
    dec_vector = np.copy(vector)
    dec_vector = np.sort(dec_vector)
    dec_vector[:] = dec_vector[::-1]
    # cumulative summation
    cusum = np.copy(vector)
    cusum[0] = dec_vector[0]
    for i in range(1, size):
        cusum[i] = cusum[i - 1] + dec_vector[i]
    # get the number of positive nonzero elements in optimal sol'n
    rho = -1
    for i in range(0, size):
        if (dec_vector[i] * (i + 1)) > cusum[i] - s:
            rho = i
    # compute lagrange multiplier
    return (cusum[rho] - s) / (rho + 1)
