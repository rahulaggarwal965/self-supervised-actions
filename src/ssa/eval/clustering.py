from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def nmi_ari(codes, labels) -> dict:
    """NMI and ARI between discovered code ids and ground-truth action labels."""
    codes = [int(c) for c in codes]
    labels = [int(x) for x in labels]
    return {
        "nmi": float(normalized_mutual_info_score(labels, codes)),
        "ari": float(adjusted_rand_score(labels, codes)),
    }
