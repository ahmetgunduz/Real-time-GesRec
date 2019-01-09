def get_mean(norm_value=255):
    return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]


def get_std(norm_value=255):
    return [
        38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value
    ]
