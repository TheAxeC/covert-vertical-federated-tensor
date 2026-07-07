"""Rebuild the paper's artifacts + map the HPC feature extraction.

COVERT's solver comparison is light and runs anywhere; the REAL features are extracted on the cluster
(HCC backbone embeddings + MIMIC-IV-ECG) and are governed / credentialed, so they are not in the repo.
This entry point runs the full pipeline on the synthetic stand-ins (end-to-end, no data needed); on the
real arms it is the same `experiments.run` with a feature `.npz` in place. Run from this directory.

    python paper.py                                        # synthetic end-to-end (run.py --dataset synth_mimic)
    python -m experiments.run --dataset hcc --ablations    # real HCC (needs COVERT_HCC_FEATURES=...npz)
    python -m experiments.run --dataset mimic --ablations  # real MIMIC (needs COVERT_MIMIC_FEATURES=...npz)

`experiments.run` emits the manuscript `\renewcommand` macro block. Full HPC feature extraction (governed
data) is one command: `bash hpc/run_all.sh` submits the from-scratch HCC imaging chain (DINO SSL -> decoder
-> finetune -> OOF embed, per modality group dwi/t1iop) + the MIMIC job (-> hcc_embeddings.npz +
mimic_features.npz). See the README "Reproduce" section for the prerequisites and the stage-by-stage map.
"""
import os
import subprocess
import sys


def main():
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.abspath(__file__)))
    print("== COVERT vs baselines on synthetic stand-ins (end-to-end) ==", flush=True)
    subprocess.run([sys.executable, "-m", "experiments.run", "--dataset", "synth_mimic"], check=True, env=env)
    print("\nDONE (synthetic). Real arms: extract the features on the cluster (see this file's header), "
          "then `python -m experiments.run --dataset {hcc,mimic} --ablations`.")


if __name__ == "__main__":
    main()
