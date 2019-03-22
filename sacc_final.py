## Created on Thu Apr 13 15:07:28 2017
## By Jeroen van Paridon
## Jeroen.vanParidon@mpi.nl


## Created on Tue Jan 31 09:58:26 2017
## By Jeroen van Paridon
## Jeroen.vanParidon@mpi.nl


## Created on Fri Oct 28 11:42:59 2016
## By Jeroen van Paridon
## Jeroen.vanParidon@mpi.nl

from mvpa2.suite import *
from pprocess import get_number_of_cores
from time import time
from os import getpid

# enter list of subject numbers for analysis
subject_list = [2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22]
#subject_list = [22, 18, 15, 12, 9, 5]
#subject_list = [21, 17, 14, 11, 8, 4]
#subject_list = [22]

cores = get_number_of_cores()


# loop over subject in subject list
for sub in subject_list:
    
    t0 = time()
    
    # compose subject name variable (not important, just useful later on)
    subject = 'S' + str(sub)
    
    # print which subject's data is currently being read in (not important, but can be useful to see if there are problems in this step)
    print('subject: ' + subject)
    
    #data_path = '/srv/clusterfs/pol/bird_fmri/up_down_48/' + subject + '/'
    data_path = 'Bird_MRI/converted/' + subject + '/'
    #data_path = 'up_down_48/' + subject + '/'
    attr_saccades_fname = data_path + 'conditions/' + subject + '_Looking_Direction.txt'
    attr_words_fname = 'up_down_48/Word_Up_Down_48.txt'
    #attr_fname = 'to_cluster_v3/Word_Up_Down.txt'
    attr_words = SampleAttributes(attr_words_fname)
    attr_saccades = SampleAttributes(attr_saccades_fname)
    
    
    
    
    
    dss_saccades = [fmri_dataset((data_path + ('glm_v4/spmT_%04d.nii' % i)), targets = attr_saccades.targets[(i - 49)], mask = (data_path + subject + '_gray_white_CSF_mask.nii'), chunks = attr_saccades.chunks[(i - 49)]) for i in range(49, 120)]
    #dss_saccades = [fmri_dataset((data_path + ('glm_v4/beta_%04d.nii' % i)), targets = attr_saccades.targets[(i - 109)], mask = (data_path + subject + '_gray_white_CSF_mask.nii'), chunks = attr_saccades.chunks[(i - 109)]) for i in range(109, 180)]
    
    #ds_words = vstack(dss_words)
    #ds_words.a.update(dss_words[0].a)
    
    ds_saccades = vstack(dss_saccades)
    ds_saccades.a.update(dss_saccades[0].a)
    
    hdr = ds_saccades.a.imghdr
    
    #print(ds_words.sa.targets)
    print(ds_saccades.sa.targets)
    #shuffle(ds)
    #print(ds.sa.targets)
    print('\nprocess ID is ' + str(getpid()) + '\n')
    print('Infs:')
    print(np.isinf(ds_saccades.samples).shape)
    print('NaNs:')
    print(np.isnan(ds_saccades.samples).shape)
    ds_saccades.samples = np.nan_to_num(ds_saccades.samples)
    zscore(ds_saccades)
    #poly_detrend(ds, polyord=1, chunks_attr='chunks')
    #ds_words = ds_words[ds_words.sa.targets != 'none']
    
    '''
    # perform some magic to balance the saccades dataset mess
    ds_h1 = ds_saccades[ds_saccades.sa.targets == 'up']
    ds_h2 = ds_saccades[ds_saccades.sa.targets == 'down']
    if len(ds_h1.sa.targets) > len(ds_h2.sa.targets):
        ds_i1 = random.sample(range(len(ds_h1.sa.targets)), (len(ds_h1.sa.targets) - len(ds_h2.sa.targets)))
        ds_h1.sa.chunks = [i for i in range(1,(len(ds_h1.sa.targets) + 1))]
        for i in ds_i1:
            ds_h1 = ds_h1[ds_h1.sa.chunks != i]
    elif len(ds_h2.sa.targets) > len(ds_h1.sa.targets):
        ds_i2 = random.sample(range(len(ds_h2.sa.targets)), (len(ds_h2.sa.targets) - len(ds_h1.sa.targets)))
        ds_h2.sa.chunks = [i for i in range(1,(len(ds_h2.sa.targets) + 1))]
        for i in ds_i2:
            ds_h2 = ds_h2[ds_h2.sa.chunks != i]
    ds_saccades = vstack([ds_h1, ds_h2])
    ds_saccades.a.update(dss_saccades[0].a)
    ds_saccades.sa.chunks = [i for i in range(1,(len(ds_h1.sa.targets) + len(ds_h2.sa.targets) + 1))]
    print(ds_saccades.sa.targets)
    print(ds_saccades.sa.chunks)
    '''
    
    # perform some magic to balance the saccades dataset mess AND fix the permutation problem
    ds_h1 = ds_saccades[ds_saccades.sa.targets == 'up']
    ds_h2 = ds_saccades[ds_saccades.sa.targets == 'down']
    ds_h1.sa.chunks = [i for i in range(1, len(ds_h1.sa.targets) + 1)]
    ds_h2.sa.chunks = [i for i in range(1, len(ds_h2.sa.targets) + 1)]
    ds_h1 = ds_h1[ds_h1.sa.chunks < 31]
    ds_h2 = ds_h2[ds_h2.sa.chunks < 31]
    ds_saccades = vstack([ds_h1, ds_h2])
    ds_saccades.a.update(dss_saccades[0].a)
    print(ds_saccades.sa.targets)
    print(len(ds_saccades.sa.targets))
    
    #zscore(ds_words)
    #zscore(ds_saccades)
    
    ds = ds_saccades
    ds.a.update(ds_saccades.a)
    
    print(ds.sa.targets)
    print(len(ds.sa.targets))
    
    #ds.sa.chunks = [i for i in range(1, len(ds.sa.targets) + 1)]
    
    # initialize random number generator with same seed for each participant to generate reproducible permutations
    #randomizer = np.random.RandomState(seed = 160517)
    #randomizer = np.random.RandomState(seed = 131117)
    randomizer = np.random.RandomState(seed = 281117)
    
    # tell PyMVPA to report on classifier progress and errors
    #debug.active += ["SVS", "SLC"]
    
    
    for i in range(50):
        i += 50
        t1 = time()
        
        # use linear kernel support vector machine
        clf = LinearCSVMC()

        # specify crossvalidation scheme
        #splt = FactorialPartitioner(NFoldPartitioner(cvtype = 1, attr = 'chunks', count=30), attr = 'targets', count=60, selection_strategy)
        splt = NFoldPartitioner(cvtype=1, attr='chunks')
        #splt = FactorialPartitioner(NGroupPartitioner(ngroups = 6, attr = 'chunks'), attr = 'targets')
        permutator = AttributePermutator('targets', limit={'partitions': 1}, count=1, rng=randomizer)
        
        '''
        # little bit of code to print generated partitions (to test folding scheme)
        q = 0
        for i in splt.generate(ds):
            q += 1
            print('set: ' + str(q))
            print(len(i.sa.partitions))
            for k in range(len(i.sa.partitions)):
                print(str(i.sa.chunks[k]) + ' ' + str(i.sa.partitions[k]) + ' ' + str(i.sa.targets[k]))
        print('total sets: ' + str(q))
        '''
        
        radius = 3
            
        null_cv = CrossValidation(clf, ChainNode([splt, permutator], space=splt.get_space()), postproc=mean_sample(), errorfx=mean_match_accuracy)
        
        # set the searchlight procedure to use previously defined crossvalidation and radius, use half of available cores (actually all of them, because we don't want to use the hyperthreading)
        sl = sphere_searchlight(null_cv, radius = radius, nproc = int(cores / 2))
        
        # actually run the searchlight
        sl_map = sl(ds)
        
        # transform the accuracy map to nifti format
        permutation_nifti = map2nifti(sl_map, imghdr = hdr)
        
        # write accuracies to file
        #permutation_nifti.to_filename('output_searchlight_test/grouplevel_cc_smoothed/permuted_crossclassify_saccades_48_3rad_' + subject + '_' + str(i) + '_50fold_fixed.nii')
        #permutation_nifti.to_filename('output_searchlight_test/grouplevel_cc_v2/real_saccades_60_3rad_' + subject + '_9fold_zscored_pointtrials.nii')
        permutation_nifti.to_filename('final-saccades/permutation_' + str(i) + '_saccades_60_3rad_' + subject + '_30fold_zscored_pointtrials.nii')
        
        t2 = time()
        t = t2 - t1
        print('searchlight ran for ' + str(int(float(t) / 60.0)) + ' min')
