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
        self.glm_maps = False
        self.hdr = False
        self.acc_maps = False
        self.permuted_acc_maps = False
        self.cluster_map = False
        self.working_dir = working_dir
        self.subject = False

    def run_searchlight(self, cv):
        sl = sphere_searchlight(cv, radius=self.sl_radius, nproc=cpu_count)
        return map2nifti(sl(self.glm_maps), imghdr=self.hdr)

    def load_glm_maps(self, data_dir, subject):
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
        self.glm_maps = ds
        self.subject = subject

    def classify(self):
        if not self.glm_maps:
            logging.error('No dataset loaded')

        # specify crossvalidation scheme
        cv = CrossValidation(self.clf, self.splt, postproc=mean_sample(), errorfx=mean_match_accuracy)

        # set up write path
        write_dir = os.path.join(self.working_dir, self.subject)
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)

        # run searchlight and write accuracy map to file
        acc_map = self.run_searchlight(cv)
        acc_map.sa.chunks = [self.subject]
        acc_map.to_filename(os.path.join(write_dir, 'acc_map.nii'))
        self.acc_maps += [acc_map]

    def permuted_classify(self, seed, n_permutations=100):
        if not self.glm_maps:
            logging.error('No dataset loaded')

        # set up write path
        write_dir = os.path.join(self.working_dir, self.subject)
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)

        # run permuted searchlight and write accuracy maps to file in loop
        self.permuted_acc_maps = []
        for i in range(n_permutations):

            # set up permutator with seed
            permutator = AttributePermutator('targets', limit={'partitions': 1}, count=1, rng=seed * i)

            # specify crossvalidation scheme including label permutation
            cv = CrossValidation(self.clf, ChainNode([self.splt, permutator], space=self.splt.get_space()),
                                 postproc=mean_sample(), errorfx=mean_match_accuracy)

            permuted_acc_map = self.run_searchlight(cv)
            permuted_acc_map.sa.chunks = [self.subject]
            permuted_acc_map.to_filename(os.path.join(write_dir, f'permuted_acc_map_{i}.nii'))
            self.permuted_acc_maps += [permuted_acc_map]

    def load_acc_maps(self, subjects):
        for subject in subjects:
            self.acc_maps += [fmri_dataset(os.path.join(self.working_dir, subject, 'acc_map.nii'), chunks=[subject])]
            self.permuted_acc_maps += [fmri_dataset(os.path.join(self.working_dir, subject, f'permuted_acc_map_{i}.nii'), chunks=[subject]) for i in range(100)]

    def cluster_inference(self, n_bootstrap=1e5):
        if not self.acc_maps:
            logging.error('No accuracy map loaded')
        elif not self.permuted_acc_maps:
            logging.error('No permuted labels accuracy map loaded')

        self.acc_maps = vstack(self.acc_maps)
        self.permuted_acc_maps = vstack(self.permuted_acc_maps)

        feature_thresh_prob = .001
        fwe_rate = .05
        multicomp_correction = 'fdr_bh'
        bootstrap = int(n_bootstrap)

        # use n_blocks = 1000 to reduce memory load when using larger bootstrap sample?
        cluster = GroupClusterThreshold(n_bootstrap=bootstrap,
                                        feature_thresh_prob=feature_thresh_prob,
                                        chunk_attr='chunks',
                                        n_blocks=1000,
                                        n_proc=1,
                                        fwe_rate=fwe_rate,
                                        multicomp_correction=multicomp_correction)

        cluster.train(self.permuted_acc_maps)
        cluster_map = cluster(self.acc_maps)
        cluster_map.samples = cluster_map.fa.clusters_fwe_thresh
        # write clusters to file
        cluster_map = map2nifti(cluster_map, imghdr=self.hdr)
        cluster_map.to_filename(os.path.join(self.working_dir, 'cluster_map.nii'))
        self.cluster_map = cluster_map

    def single_subject(self, data_dir, subject, seed):
        self.load_glm_maps(data_dir, subject)
        self.classify()
        self.permuted_classify(seed)

    def permuted_inference(self, subjects, n_bootstrap=1e5):
        self.load_acc_maps(subjects)
        self.cluster_inference(n_bootstrap)

    def whole_bird(self, data_dir, subjects, seed):
        for subject in subjects:
            self.single_subject(data_dir, subject, seed)
        self.permuted_inference()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='generate accuracy maps and permuted accuracy maps for a single subject')
    argparser.add_argument('--subject', default='S4')
    argparser.add_argument('--data_dir', default='/home/jerpar/Bird_MRI/converted/')
    argparser.add_argument('--working_dir', default='./')
    args = argparser.parse_args()

    project = Bird(args.working_dir)
    project.single_subject(args.data_dir, args.subject, seed=7)
