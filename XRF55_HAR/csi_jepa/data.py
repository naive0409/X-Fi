"""WiFi-only dataset for the csi_jepa base (PROJECT_LOG §9 Phase A1).

Mirrors XRF55_Dataset.py's directory convention and label rule but globs the WiFi modality
ONLY — WiFi-only IO halves per-sample bytes vs the triple-modality loader (~2.2 MB vs
~4.5 MB). Returns raw numpy arrays (trap 9: NO normalization in the dataloader; any
standardization lives inside the model as buffers, config-gated).
"""
import glob
import os

import numpy as np
import torch


class WiFiOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scene='all', is_train=True):
        super(WiFiOnlyDataset, self).__init__()
        scenes = ['Scene1', 'Scene2', 'Scene3', 'Scene4'] if scene == 'all' else scene
        split = 'train_data' if is_train else 'test_data'
        self.wifi_name_list = []
        for scene_sub in scenes:
            sub_list = glob.glob(os.path.join(root_dir, split, 'WiFi', scene_sub, scene_sub, '*.npy'))
            sub_list.sort()
            self.wifi_name_list += sub_list

    def __len__(self):
        return len(self.wifi_name_list)

    def __getitem__(self, idx):
        wifi_file_name = self.wifi_name_list[idx]
        wifi_data = np.load(wifi_file_name)
        # identical label rule to XRF55_Dataset.py:47 — basename field 2 minus 1
        label = int(os.path.basename(os.path.normpath(wifi_file_name)).split('_')[1]) - 1
        return wifi_data, label


def wifi_collate(batch):
    "Stack raw WiFi (B, 270, 1000) float32 + labels; no normalization here (trap 9)."
    wifi = torch.FloatTensor(np.array([t[0] for t in batch]))
    labels = torch.LongTensor([int(t[1]) for t in batch])
    return wifi, labels
