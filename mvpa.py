# -*- coding: future_fstrings -*-
import numpy as np
import logging
import argparse
import psutil
import os
from mvpa2.suite import *
logging.basicConfig(format='[{levelname}] {message}', style='{', level=logging.INFO)
cpu_count = psutil.cpu_count(logical=False)  # logical=False to count only physical cores
debug.active += ['SVS', 'SLC']  # enable PyMVPA searchlight diagnostics


class Bird(object):

    def __init__(self, working_dir):
        # set classifier and partitioner to defaults; these can be changed before starting classification
        self.clf = LinearCSVMC()
        # self.clf = GNB()
        # split data 80/20 (training/test) for stable decoding estimates cf. https://doi.org/10.1016/j.neuroimage.2016.10.038
        self.splt = NFoldPartitioner(cvtype=.2, attr='chunks', selection_strategy='random', count=10)
        self.sl_radius = 3
        self.ds = False
        self.hdr = False
        self.acc_map = False
        self.permuted_acc_maps = False
        self.cluster_map = False
        self.working_dir = working_dir
        self.subject = False

    def run_searchlight(self, cv):
        sl = sphere_searchlight(cv, radius=self.sl_radius, nproc=cpu_count)
        return map2nifti(sl(self.ds), imghdr=self.hdr)

    def classify(self):
        if not self.ds:
            logging.error('No dataset loaded')

        # specify crossvalidation scheme
        cv = CrossValidation(self.clf, self.splt, postproc=mean_sample(), errorfx=mean_match_accuracy)

        # set up write path
        write_dir = os.path.join(self.working_dir, self.subject)
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)

        # run searchlight and write accuracy map to file
        self.acc_map = self.run_searchlight(cv)
        self.acc_map.to_filename(os.path.join(write_dir, 'acc_map.nii'))

    def permuted_classify(self, seed, n_permutations=100):
        if not self.ds:
            logging.error('No dataset loaded')

        # set up write path
        write_dir = os.path.join(self.working_dir, self.subject)
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)

        # run permuted searchlight and write accuracy maps to file in loop
        self.permuted_acc_maps = []
        for i in range(n_permutations):

            # set up permutator with seed
            permutator = AttributePermutator('targets', limit={'partitions': 1}, count=1, rng=i * seed)

            # specify crossvalidation scheme including label permutation
            cv = CrossValidation(self.clf, ChainNode([self.splt, permutator], space=self.splt.get_space()),
                                 postproc=mean_sample(), errorfx=mean_match_accuracy)

            self.permuted_acc_maps += [self.run_searchlight(cv)]
            self.permuted_acc_maps[i].to_filename(os.path.join(write_dir, f'permuted_acc_map_{i}.nii'))

    def cluster_inference(self):
        if not self.acc_map:
            logging.error('No accuracy map loaded')
        elif not self.permuted_acc_maps:
            logging.error('No permuted labels accuracy map loaded')
        # TODO implement group clustering procedure
        # TODO write cluster map to nifti

    def load_dataset(self, data_dir, subject):
        # TODO load data
        path = os.path.join(data_dir, subject)
        condition = 'words'

        if condition == 'saccades':
            attr_saccades = SampleAttributes(os.path.join(data_dir, 'conditions', f'{subject}_Looking_Direction.txt'))
            dss = [fmri_dataset(os.path.join(path, f'glm_v4/spmT_{i:04}.nii'),
                                targets=attr_saccades.targets[(i - 49)],
                                mask=(os.path.join(path, f'{subject}_gray_white_CSF_mask.nii')),
                                chunks=attr_saccades.chunks[(i - 49)])
                   for i in range(49, 120)]

        elif condition == 'words':
            attr_words = SampleAttributes('/home/jerpar/up_down_48/Word_Up_Down_48.txt')
            dss = [fmri_dataset(os.path.join(path, f'glm_v4/spmT_{i + 1:04}.nii'),
                                targets=attr_words.targets[i],
                                mask=(os.path.join(path, f'{subject}_gray_white_CSF_mask.nii')),
                                chunks=attr_words.chunks[i])
                   for i in range(48)]

        ds = vstack(dss)  # stack list of datasets into single multivolume dataset
        ds.samples = np.nan_to_num(ds.samples)  # reset NaNs to zero?
        zscore(ds)

        # do a little chunking dance
        ds_h1 = ds[ds.sa.targets == 'up']
        ds_h2 = ds[ds.sa.targets == 'down']
        ds_h1.sa.chunks = [i for i in range(1, len(ds_h1.sa.targets) + 1)]
        ds_h2.sa.chunks = [i for i in range(1, len(ds_h2.sa.targets) + 1)]
        # if saccades, reject extra volumes to balance dataset
        if condition == 'saccades':
            ds_h1 = ds_h1[ds_h1.sa.chunks < 31]
            ds_h2 = ds_h2[ds_h2.sa.chunks < 31]
        ds = vstack([ds_h1, ds_h2])

        ds.a.update(dss[0].a)  # get stacked dataset attributes from volume one in dataset list
        self.hdr = ds.a.imghdr  # store imghdr for later
        self.ds = ds
        self.subject = subject

    def single_subject(self, data_dir, subject, seed):
        # TODO implement generating of all maps for single subject
        self.load_dataset(data_dir, subject)
        self.classify()
        self.permuted_classify(seed)

    def permuted_inference(self, data_dir, subjects, seed):
        # TODO implement full dataset processing
        self.load_dataset(data_dir, subjects)
        self.classify()
        self.permuted_classify(seed)
        self.cluster_inference()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='generate accuracy maps and permuted accuracy maps for a single subject')
    argparser.add_argument('--subject', default='S4')
    argparser.add_argument('--data_dir', default='/home/jerpar/Bird_MRI/converted/')
    argparser.add_argument('--working_dir', default='./')
    args = argparser.parse_args()

    project = Bird(args.working_dir)
    project.single_subject(args.data_dir, args.subject, seed=7)
