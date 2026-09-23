"""Create the XRF55 train/test split as a pure symlink tree, without touching the raw data.

Role in the project (JEPAREADME §6, PROJECT_LOG §3):
    X-Fi's XRF55_Dataset.py expects the layout
        <root>/train_data|test_data/{RFID,WiFi,mmWave}/SceneN/SceneN/*.npy
    but the raw download is organized per scene without a split:
        <raw>/SceneN/SceneN/{RFID,WiFi,mmWave}/<subject>_<action>_<trial>.npy
    This script builds <out>/{train_data,test_data}/... as symlinks into <raw>, using the
    official XRF55 split (aiotgroup/XRF55-repo split_train_test.py): of the 20 trials per
    action per subject, trials 01-14 go to train and 15-20 to test (7:3). X-Fi paper §5.1
    states they follow "the original split setting" of XRF55, so this is the matching
    protocol for reproducing baseline A.

The raw data location is fixed and must never be moved/copied; symlinks point through the
repo's data/XRF55_Dataset symlink so only one indirection needs updating if the drive moves.
Run from XRF55_HAR/:  python make_xrf55_split.py
"""
import argparse
import os

SCENES = ["Scene1", "Scene2", "Scene3", "Scene4"]
MODALITIES = ["RFID", "WiFi", "mmWave"]  # order matches XRF55_Dataset.py replace('RFID', ...)


def main():
    parser = argparse.ArgumentParser("Build XRF55 train/test split as symlink tree")
    parser.add_argument("--raw", type=str, default="../data/XRF55_Dataset",
                        help="raw XRF55 root (SceneN/SceneN/{modality}/*.npy); may be a symlink")
    parser.add_argument("--out", type=str, default="../data/XRF55_Dataset_split",
                        help="output split root; created if missing, existing links are kept")
    parser.add_argument("--train-trials", type=int, default=14,
                        help="trials with number <= this are train, the rest test (official: 14)")
    args = parser.parse_args()

    raw_root = os.path.abspath(args.raw)
    out_root = os.path.abspath(args.out)
    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"raw dataset root not found: {raw_root}")

    n_train, n_test, n_skip = 0, 0, 0
    for scene in SCENES:
        for mod in MODALITIES:
            src_dir = os.path.join(raw_root, scene, scene, mod)
            if not os.path.isdir(src_dir):
                raise FileNotFoundError(f"missing modality dir: {src_dir}")
            for fname in sorted(os.listdir(src_dir)):
                if not fname.endswith(".npy"):
                    continue
                trial = int(os.path.splitext(fname)[0].split("_")[2])  # <subject>_<action>_<trial>
                split = "train_data" if trial <= args.train_trials else "test_data"
                dst_dir = os.path.join(out_root, split, mod, scene, scene)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, fname)
                if os.path.lexists(dst):
                    n_skip += 1
                    continue
                # absolute target, resolved through the raw root (follows data/XRF55_Dataset symlink)
                os.symlink(os.path.join(src_dir, fname), dst)
                if split == "train_data":
                    n_train += 1
                else:
                    n_test += 1

    print(f"symlinks created: train={n_train}, test={n_test}, already existed (skipped)={n_skip}")
    print(f"split root: {out_root}")
    print("use it as:  python run.py --dataset ../data/XRF55_Dataset_split")


if __name__ == "__main__":
    main()
