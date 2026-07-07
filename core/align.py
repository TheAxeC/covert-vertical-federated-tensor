"""
Cohort alignment / private set intersection (the protocol step that produces the aligned
per-party feature matrices). On the HPC this runs over real subject_ids before feature export;
once features are saved aligned (same N rows), downstream loading is a no-op verification.

private_set_intersection here is the *plaintext* reference (for building the aligned cohort at
data-prep time). The DEPLOYED protocol replaces it with a cryptographic PSI (e.g. ECDH-PSI /
Bonawitz-style) so no party learns the other's non-intersecting ids - but the resulting aligned
cohort is identical, so experiments are unaffected.
"""
import numpy as np


def private_set_intersection(ids_per_party):
    """ids_per_party: list of 1-D arrays of patient ids held by each party.
    Returns (common_ids_sorted, [index_array_per_party]) selecting the shared cohort in order."""
    common = set(map(tuple if False else (lambda a: a), ids_per_party[0].tolist()))
    common = set(ids_per_party[0].tolist())
    for ids in ids_per_party[1:]:
        common &= set(ids.tolist())
    common_sorted = np.array(sorted(common))
    idx = []
    for ids in ids_per_party:
        pos = {v: i for i, v in enumerate(ids.tolist())}
        idx.append(np.array([pos[v] for v in common_sorted]))
    return common_sorted, idx


def verify_aligned(data):
    """Sanity-check that a loaded dataset's parties are row-aligned on one cohort."""
    n = data['y'].shape[0]
    for X, nm in zip(data['parties'], data['party_names']):
        assert X.shape[0] == n, f"party {nm} has {X.shape[0]} rows != cohort {n}"
    assert data['groups'].shape[0] == n
    return True
