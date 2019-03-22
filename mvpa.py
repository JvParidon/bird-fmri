# -*- coding: future_fstrings -*-
import numpy as np
import logging
import argparse
import psutil
import os
from mvpa2.suite import *
logging.basicConfig(format='[{levelname}] {message}', style='{', level=logging.INFO)
debug.active += ['SVS', 'SLC']
cpu_count = psutil.cpu_count(logical=False)


class Bird(object):

    def __init__(self, prefix, seed):
        self.clf = LinearCSVMC()
        self.splt = NFoldPartitioner(cvtype=1, attr='chunks')
        self.permutator = AttributePermutator('targets', limit={'partitions': 1}, count=1,
                                              rng=np.random.RandomState(seed))
        self.sl_radius = 3
        self.ds = False
        self.hdr = False
        self.acc_map = False
        self.permuted_acc_maps = False
        self.cluster_map = False
        self.prefix = prefix
        self.subject = False

    def run_searchlight(self, cv):
        sl = sphere_searchlight(cv, radius=self.sl_radius, nproc=cpu_count)
        return map2nifti(sl(self.ds), self.hdr)

    def classify(self):
        if not self.ds:
            logging.error('No dataset loaded')

        # specify crossvalidation scheme
        cv = CrossValidation(clf, splt, postproc=mean_sample(), errorfx=mean_match_accuracy)

        # run searchlight and write accuracy map to file
        self.acc_map = self.run_searchlight(cv)
        self.acc_map.to_filename(f'{self.prefix}/{self.subject}/acc_map.nii')

    def permuted_classify(self, n_permutations=100):
        if not self.ds:
            logging.error('No dataset loaded')

        # specify crossvalidation scheme including label permutation
        cv = CrossValidation(self.clf, ChainNode([self.splt, self.permutator], space=self.splt.get_space()),
                             postproc=mean_sample(), errorfx=mean_match_accuracy)

        # run searchlight and write accuracy maps to file
        self.permuted_acc_maps = []
        for i in range(n_permutations):
            self.permuted_acc_maps += [self.run_searchlight(cv)]
            self.permuted_acc_maps[i].to_filename(f'{self.prefix}/{self.subject}/permuted_acc_map_{i}.nii')

    def cluster_inference(self):
        if not self.acc_map:
            logging.error('No accuracy map loaded')
        elif not self.permuted_acc_maps:
            logging.error('No permuted labels accuracy map loaded')
        # TODO implement group clustering procedure
        # TODO write cluster map to nifti

    def load_dataset(self, directory, subject):
        # TODO load data
        path = os.path.join(directory, subject)
        dss = [fmri_dataset(os.path.join(path, f'glm_v4/spmT_{i}.nii'),
                            targets=attr_saccades.targets[(i - 49)],
                            mask=(path + subject + '_gray_white_CSF_mask.nii'),
                            chunks=attr_saccades.chunks[(i - 49)])
               for i in range(49, 120)]

        ds_h1 = ds[ds.sa.targets == 'up']
        ds_h2 = ds[ds.sa.targets == 'down']
        ds_h1.sa.chunks = [i for i in range(1, len(ds_h1.sa.targets) + 1)]
        ds_h2.sa.chunks = [i for i in range(1, len(ds_h2.sa.targets) + 1)]
        ds = vstack([ds_h1, ds_h2])
        ds.a.update(dss[0].a)

        self.ds = ds

    def single_subject(self, directory, subject):
        # TODO implement generating of all maps for single subject
        self.load_dataset(directory, subject)
        self.classify()
        self.permuted_classify()

    def permuted_inference(self, dataset_dir, subjects):
        # TODO implement full dataset processing
        self.load_dataset(dataset_dir, subjects)
        self.classify()
        self.permuted_classify()
        self.cluster_inference()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='generate accuracy maps and permuted accuracy maps for a single subject')
    argparser.add_argument('--subject')
    argparser.add_argument('--directory')
    args = argparser.parse_args()

    project = Bird(prefix='Bird1', seed=7)
    Bird.single_subject(args.directory, args.subject)
