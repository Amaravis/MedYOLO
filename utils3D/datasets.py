"""
Dataloaders and dataset utils for nifti datasets for YOLO3D
"""

# standard library imports
import nibabel as nib
import numpy as np
from pathlib import Path
import glob
import os
from multiprocessing.pool import Pool
from tqdm import tqdm
from itertools import repeat
from typing import List
import torch
from torch.utils.data import Dataset
from scipy.ndimage import binary_dilation, binary_erosion

# 2D YOLO imports
from utils.torch_utils import torch_distributed_zero_first
from utils.datasets import InfiniteDataLoader, get_hash

# 3D YOLO imports
from utils3D.general import zxyzxy2zxydwhn, zxydwhn2zxyzxy
from utils3D.augmentations import tensor_cutout, random_zoom


# Configuration
IMG_FORMATS = ['nii', 'nii.gz']  # acceptable image suffixes, note nii.gz compatible by checking for presence of 'nii' in -2 place
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads
default_size = 350 # edge length for testing


def file_lister_train(parent_dir: List[str], prefix=''):
    """Takes a parent directory or list of parent directories and
    looks for files within those directories.  Output organized to fit
    YOLO training requirements.

    Args:
        parent_dir (List[str] or str): Folders to be searched.  Text files allowed.
        prefix (str, optional): Prefix for error messages. Defaults to ''.

    Raises:
        Exception: If parent_dir is neither a directory nor file.

    Returns:
        file_list (List[str]): a list of paths to the files found.
        p (pathlib.PosixPath): the path to the parent directory, for caching purposes
    """

    file_list = []
    for p in parent_dir if isinstance(parent_dir, list) else [parent_dir]:
        p = Path(p)
        if p.is_dir():  # dir
            file_list += glob.glob(str(p / '**' / '*.*'), recursive=True)
        elif p.is_file():  # file
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                file_list += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                # file_list += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        else:
            raise Exception(f'{prefix}{p} does not exist')
    return file_list, p


def file_lister_detect(parent_dir: str):
    """Takes a parent directory and looks for files within those directories.
    Output organized to fit YOLO inference requirements.

    Args:
        parent_dir (str): parent folder to search for files.

    Raises:
        Exception: if parent_dir is not a file, directory, or glob search pattern.

    Returns:
        files (List[str]): a list of paths to the files found.
    """
    p = str(Path(parent_dir).resolve())
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return files


class LoadNiftis(Dataset):
    """YOLO3D Pytorch Dataset for inference."""
    def __init__(self, path: str, img_size=default_size, stride=32):
        """Initialization for the inference Dataset

        Args:
            path (str): parent directory for the Dataset's files
            img_size (int, optional): edge length for the cube input will be reshaped to. Defaults to default_size (currently 350).
            stride (int, optional): model stride, used for resizing and augmentation, currently unimplemented. Defaults to 32.
        """
        
        # Find files in the given path and filter to leave only .nii and .nii.gz files in the list
        files = file_lister_detect(path)
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS or x.split('.')[-2].lower() in IMG_FORMATS]

        self.nf = len(images)
        self.files = images
        self.img_size = img_size
        self.stride = stride

        assert self.nf > 0, f'No images found in {path}. Supported formats are: {IMG_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # Iterate through the list of files
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read current image
        self.count += 1
        # img0, affine = open_nifti(path)
        img0, _ = open_nifti(path)
        assert img0 is not None, 'Image Not Found ' + path
        print(f'\nimage {self.count}/{self.nf} {path}: ', end='')

        # Reshape image to fit model requirements
        img = transpose_nifti_shape(img0)
        img = change_nifti_size(img, self.img_size)

        return path, img, img0

    def __len__(self):
        return self.nf  # number of files


class LoadNiftisAndLabels(Dataset):
    """YOLO3D Pytorch Dataset for training."""
    cache_version = 0.61  # dataset labels *.cache version

    def __init__(self, path, img_size=default_size, batch_size=4, augment=False, hyp=None, single_cls=False,
                 stride=32, pad=0.0, prefix='', 
                 sample_negatives=True,          
                 neg_ratio=1.0,                   
                 max_iou_neg=0.05,                
                 size_jitter=0.25,                
                 brain_coverage_thresh=0.35,      
                 min_neg_when_no_pos=4,           
                 brain_dilate=2,               
                 max_neg_trials=2000,             
                 return_negatives=True,mask_percentile=10):         # if True, __getitem__ returns negatives as 5th item):
        """Initialization for the training Dataset

        Args:
            path (str): parent directory for the Dataset's files
            img_size (int, optional): edge length for the cube input will be reshaped to. Defaults to default_size (currently 350).
            batch_size (int, optional): size of the batch to return. Defaults to 4.
            augment (bool, optional): determines whether data will be augmented, currently unimplemented. Defaults to False.
            hyp (Dict, optional): dictionary containing augmentation configuration hyperparameters, currently unimplemented. Defaults to None.
            stride (int, optional): model stride, used for resizing and augmentation, currently unimplemented. Defaults to 32.
            pad (float, optional): image padding, used for resizing and augmentation, currently unimplemented. Defaults to 0.0.
            prefix (str, optional): Prefix for error messages. Defaults to ''.

        Raises:
            Exception: if unable to load data in given path.
        """       
        self.img_size = img_size
        self.stride = stride
        self.path = path
        self.augment = augment
        self.hyp = hyp
        self.sample_negatives = sample_negatives
        self.neg_ratio = float(neg_ratio)
        self.max_iou_neg = float(max_iou_neg)
        self.size_jitter = float(size_jitter)
        self.brain_coverage_thresh = float(brain_coverage_thresh)
        self.min_neg_when_no_pos = int(min_neg_when_no_pos)
        self.brain_dilate = int(brain_dilate)
        self.max_neg_trials = int(max_neg_trials)
        self.return_negatives = bool(return_negatives)
        self.mask_percentile = float(mask_percentile)

        # Find files in the given path and filter to leave only .nii and .nii.gz files in the list
        try:
            f, p = file_lister_train(path, prefix)
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS or x.split('.')[-2].lower() in IMG_FORMATS) # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}')

        # Check cache and find the labels
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.img_files)  # same hash
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels.'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        # nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, label in enumerate(self.labels):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        self.imgs, self.img_npy = [None] * n, [None] * n

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        """Caches dataset labels, verifies images and reads their shapes.
        See: verify_image_label function

        Args:
            path (pathlib.Path, optional): Path to write cache to. Defaults to Path('./labels.cache').
            prefix (str, optional): prefix for error messages. Defaults to ''.

        Returns:
            x (Dict): Dictionary containing the results of the image search.
        """
        x = {}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        except Exception as e:
            print(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')
        return x

    def load_nifti(self, i):
        """Reads a nifti file, converts it to a torch.tensor, and reshapes and resizes it for use in the YOLO3D model.

        Args:
            i (int): Dataset index for the nifti to be loaded

        Returns:
            im (torch.tensor): YOLO3D input tensor containing the nifti image data
            d0 (int): original image depth
            h0 (int): original image height
            w0 (int): original image width
            im.size()[1:] (List(int)): current image depth, height, and width
        """
        # loads 1 image from dataset index 'i'
        path = self.img_files[i]
        im, affine = open_nifti(path)

        # reshape im from height, width, depth to depth, height, width to make it compatible with torch convolutions
        im = transpose_nifti_shape(im)

        d0, h0, w0 = im.size()

        # resize im to self.img_size
        im = change_nifti_size(im, self.img_size)

        return im, (d0, h0, w0), im.size()[1:], affine

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """Loads niftis and converts to torch tensor to be fed as input to the model.

        Args:
            index (int): dataset index of image to be read.

        Returns:
            img (torch.tensor): image data from loaded nifti, potentially augmented
            labels_out (torch.tensor): labels corresponding to loaded nifti, with augmentation accounted for
            self.img_files[index] (str): path to the loaded nifti
            shapes (Tuple[Tuple[float]]): Tuple containing Tuples of relative shape information for original image, resized image, and padding
        """
        # Load image
        img, (d0, h0, w0), (d, h, w), _ = self.load_nifti(self.indices[index])

        # Letterbox
        # shape = (self.img_size, self.img_size, self.img_size) # not adding rectangular training yet
        # img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment) # not implemented
        # ratio = (1, 1, 1) # no letterboxing so the shape doesn't change and the ratios are all 1
        pad = (0, 0, 0) # shape not changing so not padding any side
        shapes = (d0, h0, w0), ((d/d0, h/h0, w/w0), pad)

        labels = self.labels[self.indices[index]].copy()
        nl = len(labels)  # number of labels

        if self.augment:           
            # Label transformation is done to make certain augmentations more straightforward
            if labels.size:  # normalized zxydwh to pixel zxyzxy format
                labels[:, 1:] = zxydwhn2zxyzxy(labels[:, 1:], d, w, h, pad[0], pad[1], pad[2])                    

            # random zoom
            img, labels = random_zoom(img, labels, self.hyp['max_zoom'], self.hyp['min_zoom'], self.hyp['prob_zoom'])
        
            # transformation of labels back to standard format
            if nl:
                labels[:, 1:7] = zxyzxy2zxydwhn(labels[:, 1:7], d=img.shape[1], w=img.shape[3], h=img.shape[2], clip=True, eps=1E-3)
            
            # Albumentations
            
            # HSV color-space
            
            # Flip up-down
            
            # Flip left-right
            
            # Cutouts
            img, labels = tensor_cutout(img, labels, self.hyp['cutout_params'], self.hyp['prob_cutout'])
            # update after cutout
            nl = len(labels)  # number of labels
        
        labels_out = torch.zeros((nl, 8))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        #return img, labels_out, self.img_files[self.indices[index]], shapes
        if not self.sample_negatives:
            return img, labels_out, self.img_files[self.indices[index]], shapes

        # --- RANDOM NEGATIVES FROM BRAIN MASK (no IoU) ---
        k_neg = int(nl * self.neg_ratio) if nl > 0 else int(self.min_neg_when_no_pos)
        #print(labels)
        neg_norm = self._sample_random_negatives_noiou(img, k_neg, labels[:, 1:7] if nl else None)

        neg_out = torch.zeros((len(neg_norm), 8), dtype=torch.float32)
        if len(neg_norm):
            # Put cls = -1 as a sentinel for “background”; your loss/target code should skip cls < 0.
            neg = np.concatenate([np.full((len(neg_norm), 1), 1.0, dtype=np.float32), neg_norm], axis=1)
            neg_out[:, 1:] = torch.from_numpy(neg)

        if not self.return_negatives:
            # If you’d rather append negatives to labels_out, you can:
            # labels_out = torch.cat([labels_out, neg_out], dim=0)
            return img, labels_out, self.img_files[self.indices[index]], shapes

        # return negatives as a separate tensor (preferred for custom losses / mining)
        return img, labels_out, self.img_files[self.indices[index]], shapes, neg_out


    def _brain_mask(self, img_4d):
        """
        Build a coarse brain mask (D,H,W) from image (1,D,H,W).
        Threshold at a robust low percentile of non-zero intensities, then optional smooth.
        """
        vol = img_4d[0].detach().cpu().numpy()  # (D,H,W)
        nz = vol[vol > 0]
        if nz.size == 0:
            return np.zeros_like(vol, dtype=bool)
        t = np.percentile(nz, self.mask_percentile)
        mask = vol > max(t, 0.0)
        if self.brain_dilate > 0:
            mask = binary_dilation(mask, iterations=self.brain_dilate)
            mask = binary_erosion(mask, iterations=1)
        return mask

    @staticmethod
    def collate_fn(batch):
        """Used to collate images to create the input batches"""
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn_with_negatives(batch):
        img, pos, path, shapes, neg = zip(*batch)
        imgs = torch.stack(img, 0)

        pos_cat = []
        for i, l in enumerate(pos):
            if l.numel():
                l = l.clone()
                l[:, 0] = i  # image index for targets
            pos_cat.append(l)
        pos_cat = torch.cat(pos_cat, 0) if len(pos_cat) else torch.zeros((0, 8))

        neg_cat = []
        for i, l in enumerate(neg):
            if l.numel():
                l = l.clone()
                l[:, 0] = i
            neg_cat.append(l)
        neg_cat = torch.cat(neg_cat, 0) if len(neg_cat) else torch.zeros((0, 8))

        return imgs, pos_cat, neg_cat, path, shapes

    @staticmethod
    def _zxydwhn_to_abs(lbls_n, D, H, W):
        """Convert YOLO-norm [cls,z,x,y,d,w,h] -> absolute (z,x,y,d,w,h)."""
        if lbls_n.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        a = lbls_n.copy()
        a[:, 0] *= D
        a[:, 1] *= W
        a[:, 2] *= H
        a[:, 3] *= D
        a[:, 4] *= W
        a[:, 5] *= H
        return a[:, 1:7].astype(np.float32)


    @staticmethod
    def _abs_to_zxydwhn(boxes_abs, D, H, W):
        """(z,x,y,d,w,h) absolute -> normalized YOLO order [z,x,y,d,w,h]."""
        if boxes_abs.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        b = boxes_abs.astype(np.float32).copy()
        b[:, 0] /= D; b[:, 1] /= W; b[:, 2] /= H
        b[:, 3] /= D; b[:, 4] /= W; b[:, 5] /= H
        return b

    @staticmethod
    def _integral3d(mask_bool):
        """
        3D summed area table of a boolean mask (D,H,W) -> (D+1,H+1,W+1)
        Allows O(1) box sums.
        """
        integ = mask_bool.astype(np.uint8).cumsum(0).cumsum(1).cumsum(2)
        out = np.zeros((mask_bool.shape[0]+1, mask_bool.shape[1]+1, mask_bool.shape[2]+1), dtype=np.int32)
        out[1:, 1:, 1:] = integ
        return out

    @staticmethod
    def _sum_box_integral(integ, z1, z2, y1, y2, x1, x2):
        """
        Sum of mask over [z1:z2, y1:y2, x1:x2] using integral image.
        All indices are ints, half-open ranges; assumes 0<=z1<z2<=D etc.
        """
        return (integ[z2, y2, x2] - integ[z1, y2, x2] - integ[z2, y1, x2] - integ[z2, y2, x1]
                + integ[z1, y1, x2] + integ[z1, y2, x1] + integ[z2, y1, x1] - integ[z1, y1, x1])

    
    def _sample_random_negatives_noiou(self, img, count, pos_lbls_norm=None):
        """
        Fast negative sampler: NO IoU checks, only requires 'brain coverage' in the box.
        - Draw centers from brain voxels.
        - Draw sizes from positive size distribution if available (with jitter), otherwise from a small default range.
        Returns (K,6) normalized [z,x,y,d,w,h] (no class column).
        """
        D, H, W = img.shape[1], img.shape[2], img.shape[3]
        mask = self._brain_mask(img)               # (D,H,W) bool
        if not mask.any():
            return np.zeros((0, 6), dtype=np.float32)

        integ = self._integral3d(mask)             # (D+1,H+1,W+1)
        coords = np.argwhere(mask)                 # (N, 3) as (z,y,x)
        rng = np.random.default_rng()

        # Base sizes: prefer positive sizes (absolute) with jitter; else fallback small-ish cubes
        base_sizes = None
        if pos_lbls_norm is not None and pos_lbls_norm.size:
            abs_pos = self._zxydwhn_to_abs(pos_lbls_norm, D, H, W)
            if abs_pos.shape[0] > 0:
                base_sizes = abs_pos[:, 3:6]  # (d,w,h)

        out = []
        trials = 0
        max_trials = max(self.max_neg_trials, 50 * (count + 1))

        while len(out) < count and trials < max_trials:
            trials += 1

            # --- choose size ---
            if base_sizes is not None:
                #b = base_sizes[rng.integers(0, len(base_sizes))]
                #jitter = rng.uniform(1.0 - self.size_jitter, 1.0 + self.size_jitter, size=3)
                #d, w, h = np.clip(b * jitter, 4.0, [D * 0.4, W * 0.4, H * 0.4]).astype(np.float32)
                #d, w, h = base_sizes[0].astype(np.float32)
                d, w, h = 10, 24, 24
            else:
                # default small range (1–6% of dimension), clipped to >=4 vox
                #d = float(rng.integers(max(4, int(0.01 * D)), max(6, int(0.06 * D)) + 1))
                #w = float(rng.integers(max(4, int(0.01 * W)), max(6, int(0.06 * W)) + 1))
                #h = float(rng.integers(max(4, int(0.01 * H)), max(6, int(0.06 * H)) + 1))
                d = 10
                w = 24
                h = 24

            # --- choose center from brain voxels, then clamp to keep box in-bounds ---
            zc, yc, xc = coords[rng.integers(0, len(coords))]
            z1 = int(max(0, np.round(zc - d / 2))); y1 = int(max(0, np.round(yc - h / 2))); x1 = int(max(0, np.round(xc - w / 2)))
            z2 = int(min(D, z1 + int(np.round(d)))); y2 = int(min(H, y1 + int(np.round(h)))); x2 = int(min(W, x1 + int(np.round(w))))
            if z2 - z1 < 2 or y2 - y1 < 2 or x2 - x1 < 2:
                continue

            # --- brain coverage check (avoid black edges) ---
            brain_vox = self._sum_box_integral(integ, z1, z2, y1, y2, x1, x2)
            frac = brain_vox / float((z2 - z1) * (y2 - y1) * (x2 - x1))
            if frac < self.brain_coverage_thresh:
                continue

            # accept; convert to center-size (z,x,y,d,w,h)
            z = (z1 + z2) / 2.0; y = (y1 + y2) / 2.0; x = (x1 + x2) / 2.0
            d = float(z2 - z1); h = float(y2 - y1); w = float(x2 - x1)
            out.append([z, x, y, d, w, h])

        if not out:
            return np.zeros((0, 6), dtype=np.float32)

        out = np.asarray(out, dtype=np.float32)
        # normalize to [0,1] in YOLO order [z,x,y,d,w,h]
        out[:, 0] /= D; out[:, 1] /= W; out[:, 2] /= H
        out[:, 3] /= D; out[:, 4] /= W; out[:, 5] /= H
        return out

def nifti_dataloader(path: str, imgsz: int, batch_size: int, stride: int, single_cls=False, hyp=None, augment=False, pad=0.0,
                     rank=-1, workers=8, prefix='',sample_negatives=True, return_negatives=True):
    """This is the dataloader used in the training process
    The same as that of 2D YOLO, just built around a different Dataset definition

    Args:
        path (str): path to the directory containing the training files
        imgsz (int): edge length for the cube input will be reshaped to.
        batch_size (int): size of the batch to return.
        stride (int): model stride, used for resizing and augmentation
        hyp (Dict, optional): dictionary containing augmentation configuration hyperparameters. Defaults to None.
        augment (bool, optional): whether or not augmentation should be enabled. Defaults to False.
        pad (float, optional): image padding, used for resizing and augmentation. Defaults to 0.0.
        rank (int, optional): determines whether to use distributed sampling. Defaults to -1.
        workers (int, optional): number of dataloader workers. Defaults to 8.
        prefix (str, optional): Prefix for error messages. Defaults to ''.

    Returns:
        dataloader: dataloader for training loop
        dataset: training dataset
    """
    
    with torch_distributed_zero_first(rank):
        dataset = LoadNiftisAndLabels(path, imgsz, batch_size,
                                      augment=augment,
                                      hyp=hyp,
                                      # rect=rect,  # rectangular training
                                      single_cls=single_cls,
                                      stride=stride,
                                      pad=pad,
                                      prefix=prefix,sample_negatives=sample_negatives,return_negatives=return_negatives)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = InfiniteDataLoader
    collate = LoadNiftisAndLabels.collate_fn if not getattr(dataset, "return_negatives", False) \
          else LoadNiftisAndLabels.collate_fn_with_negatives
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True, # may need to set False to resolve memory issues
                        collate_fn=collate)
    return dataloader, dataset


def img2label_paths(img_paths):
    """Defines label paths as a function of the image paths.  Filters for .nii and .nii.gz files.

    Args:
        img_paths (List[str]): list of image file paths to convert to label file paths.

    Returns:
        List[str]: list of label file paths
    """
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    label_paths = []

    # to handle both compressed and uncompressed niftis
    for x in img_paths:
        if x.endswith('.nii.gz'):
            label_paths.append(sb.join(x.rsplit(sa, 1)).rsplit('.', 2)[0] + '.txt')
        elif x.endswith('.nii'):
            label_paths.append(sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt')

    return label_paths


def verify_image_label(args):
    """Verify one image-label pair.  Works for .nii and .nii.gz files.

    Args:
        args (Tuple[str]): contains the image path, label path, and error message prefix

    Returns:
        im_file (str): path to the image file
        l (List[float]): labels corresponding to the image file
        shape (List[int]): 3D shape of the image file
        segments: Alternate representation of image shape, currently not supported but necessary for compatibility with YOLOv5 code.
        nm (int): 1 if label missing, 0 if label found
        nf (int): 1 if label found, 0 if label not found
        ne (int): 1 if label empty, 0 if label not empty
        nc (int): 1 if label corrupted and Exception found, 0 if not
        msg (str): Message returned in the event an error occurs
    """
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = np.array(nib.load(im_file).dataobj)
        shape = (im.shape[2], im.shape[0], im.shape[1]) # need to transpose to account for depth reshaping that will happen to image tensors
        # assert call may need to be reworked for non-nifti data-types or if larger minimum sizes are required
        assert (shape[0] > 9) & (shape[1] > 99) & (shape[2] > 99), f'image size {shape} < 10x100x100 voxels'
        assert im_file.split('.')[-1].lower() in IMG_FORMATS or im_file.split('.')[-2].lower() in IMG_FORMATS, f'invalid image format {im_file}'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1 # label found
            with open(lb_file) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                # segments aren't supported for simplicity
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 7, f'labels require 7 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                l = np.unique(l, axis=0)  # remove duplicate rows
                if len(l) < nl:
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 7), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 7), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def open_nifti(filepath: str):
    """Reads a nifti file and converts it to a torch tensor

    Args:
        filepath (str): Path to the nifti file

    Returns:
        nifti (torch.tensor): Tensor containing the nifti image data
        nifti_affine: affine array for the nifti
    """
    nifti = nib.load(filepath)
    nifti_affine = nifti.affine
    # nifti_array = np.array(nifti.dataobj)
    # assert nifti_array is not None, 'Image Not Found ' + filepath
    # nifti_tensor = torch.tensor(nifti_array, dtype=torch.float)
    # return nifti_tensor, nifti_affine
    nifti = np.array(nifti.dataobj)
    assert nifti is not None, 'Image Not Found ' + filepath
    nifti = torch.tensor(nifti, dtype=torch.float)
    return nifti, nifti_affine


def transpose_nifti_shape(nifti_tensor: torch.Tensor):
    """Reshapes the tensor from height, width, depth order to depth, height, width
    to make it compatible with torch convolutions.

    Args:
        nifti_tensor (torch.tensor): tensor to be reshaped

    Returns:
        nifti_tensor (torch.tensor): reshaped tensor
    """
    nifti_tensor = torch.transpose(nifti_tensor, 0, 2)
    nifti_tensor = torch.transpose(nifti_tensor, 1, 2)
    return nifti_tensor


def change_nifti_size(nifti_tensor: torch.Tensor, new_size: int):
    """Resizes a 3D tensor to a cube with edge length new_size.
    Also adds the channel dimension.

    Args:
        nifti_tensor (torch.Tensor): The tensor to be resized
        new_size (int): The edge length for the resized, cubic tensor

    Returns:
        nifti_tensor (torch.tensor): Resized, cubic tensor
    """
    # add channel dimension for compatibility with later code
    nifti_tensor = torch.unsqueeze(nifti_tensor, 0)
    # add batch dimension for functional interpolate
    nifti_tensor = torch.unsqueeze(nifti_tensor, 0)
    # resize image to a cube of size new_size
    nifti_tensor = torch.nn.functional.interpolate(nifti_tensor, size=(new_size, new_size, new_size), mode='trilinear', align_corners=False)
    # remove batch dimension for compatibility with later code
    nifti_tensor = torch.squeeze(nifti_tensor, 0)
    return nifti_tensor


def normalize_CT(imgs):
    """Normalizes 3D CTs in Hounsfield Units (+/- 1024) to within 0 and 1.

    Args:
        imgs (torch.tensor): unnormalized model input

    Returns:
        imgs (torch.tensor): normalized model input
    """
    imgs = (imgs + 1024.) / 2048.0  # int to float32, -1024-1024 to 0.0-1.0
    return imgs


def normalize_MR(imgs):
    """Volume normalizes 3D MR images to mean 0 and standard deviation 1.

    Args:
        imgs (torch.tensor): unnormalized model input

    Returns:
        imgs (torch.tensor): normalized model input
    """
    means = torch.mean(imgs, dim=[1,2,3,4], keepdim=True)
    std_devs = torch.std(imgs, dim=[1,2,3,4], keepdim=True)
    imgs = (imgs - means)/std_devs
    return imgs
