import math

def cosine_annealing(t, alpha_max, alpha_min, T_w, T_c):
    """Cosine annealing learning rate scheduler.
    t: current iteration
    alpha_max: maximum learning rate
    alpha_min: minimum learning rate
    T_w: warmup period
    T_c: cosine annealing period
    """
    if t < T_w:
        return t * alpha_max / T_w
    if t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos((t - T_w) * math.pi / (T_c - T_w))) * (alpha_max - alpha_min)

    return alpha_min