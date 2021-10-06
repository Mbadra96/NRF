def clamp_signal_to_spikes(e):
    if e > 0:
        return 1, 0
    elif e < 0:
        return 0, 1
    else:
        return 0, 0
