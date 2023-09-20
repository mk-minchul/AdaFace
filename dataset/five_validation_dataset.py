from torch.utils.data import Dataset
import numpy as np
import evaluate_utils
import torch

class MultipleValidationDataset(Dataset):
    def __init__(self, val_data_dict, concat_mem_file_name):
        '''
        concatenates all validation datasets from emore, for instance:
        val_data_dict = {
        'agedb_30': (agedb_30, agedb_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
        ...
        }
        agedb_30: 0
        cfp_fp: 1
        lfw: 2
        cplfw: 3
        calfw: 4
        '''
        self.dataname_to_idx ={} #{"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
        for i, key in enumerate(val_data_dict.keys()):
            print(key, " is present")
            self.dataname_to_idx[key] = i
        next_idx = 5
        self.val_data_dict = val_data_dict
        # concat all dataset
        all_imgs = []
        all_issame = []
        all_dataname = []
        key_orders = []
        print()
        for key, (imgs, issame) in val_data_dict.items():
            all_imgs.append(imgs)
            dup_issame = []  # hacky way to make the issame length same as imgs. [1, 1, 0, 0, ...]
            for same in issame:
                dup_issame.append(same)
                dup_issame.append(same)
            assert len(dup_issame) == len(imgs), f"found {len(dup_issame)} labels for {len(imgs)} imgs. Please check the following dataset: {key}"
            all_issame.append(dup_issame)
            if not key in self.dataname_to_idx.keys():
                raise Exception("error met, keys unconsistency")
            all_dataname.append([self.dataname_to_idx[key]] * len(imgs))
            
            key_orders.append(key)
        # assert is irrelevent since the switch to ordereddict but keeping it for fun. If you want to get rid of hardcode stuff remove this
        assert np.all([key in key_orders for key in ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']]), f"assert is irrelevent since the switch to ordereddict but keeping it for fun. If you want to get rid of hardcode stuff remove this"

        if isinstance(all_imgs[0], np.memmap):
            self.all_imgs = evaluate_utils.read_memmap(concat_mem_file_name)
        else:
            self.all_imgs = np.concatenate(all_imgs)

        self.all_issame = np.concatenate(all_issame)
        self.all_dataname = np.concatenate(all_dataname)

        assert len(self.all_imgs) == len(self.all_issame), f"after concatenation {len(self.all_imgs)} images found vs {len(self.all_issame)} labels,  maybe regenerate the memfiles"
        assert len(self.all_issame) == len(self.all_dataname), f"after concatenation {len(self.all_dataname)} dataname associations found vs {len(self.all_issame)} labels"

    def __getitem__(self, index):
        x_np = self.all_imgs[index].copy()
        x = torch.tensor(x_np)
        y = self.all_issame[index]
        dataname = self.all_dataname[index]

        return x, y, dataname, index

    def __len__(self):
        return len(self.all_imgs)


