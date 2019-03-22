#import os
import sys
print('\n'.join(sys.path))
import mvpa2
print(mvpa2.wtf())

from mvpa2.suite import *
from mvpa2.clfs.svm import LinearCSVMC
from pprocess import get_number_of_cores


# enter list of subject numbers for analysis
subject_list = [7]

cores = get_number_of_cores()


# loop over subject in subject list
for sub in subject_list:
    
    # compose subject name variable (not important, just useful later on)
    subject = ('S' + str(sub))
    
    print('subject: ' + subject)
    
    #data_path = '/srv/clusterfs/pol/bird_fmri/up_down_48/' + subject + '/'
    data_path = '/data/clusterfs/pol/bird_fmri/Bird1_tmaps/' + subject + '/glm_v4/'
    attr_fname = '/data/clusterfs/pol/bird_fmri/Bird1/' + 'Bird_word_decoding_stimuli.txt'
    attr = SampleAttributes(attr_fname)

    
    dss = [fmri_dataset((data_path + ('spmT_%04d.nii' % i)), targets = attr.targets[(i - 1)], mask = (data_path + 'mask.nii'), chunks = attr.chunks[(i - 1)]) for i in range(1,49)]
    #dss = [fmri_dataset((data_path + ('spmT_%04d.nii' % i)), targets = attr.targets[(i - 1)], mask = 'rSPL_mask.nii', chunks = attr.chunks[(i - 1)]) for i in range(1,49)]

    ds = vstack(dss)
    ds.a.update(dss[0].a)
    
    hdr = ds.a.imghdr
    ds = ds[ds.sa.targets != 'none']
    
    print(ds.sa.targets)
    #shuffle(ds)
    #print(ds.sa.targets)
    
    #poly_detrend(ds, polyord=1, chunks_attr='chunks')
    
    zscore(ds)
    
    print(ds.sa.targets)

    ds.sa.chunks = [i for i in range(1, len(ds.sa.targets) + 1)]

    
    # tell PyMVPA to report on classifier progress and errors
    debug.active += ["SVS", "SLC"]
    
     # use linear kernel support vector machine
    clf = LinearCSVMC()
    
    
    ds_h1 = ds[ds.sa.targets == 'up']
    ds_h2 = ds[ds.sa.targets == 'down']
    ds_h1.sa.chunks = [i for i in range(1, len(ds_h1.sa.targets) + 1)]
    ds_h2.sa.chunks = [i for i in range(1, len(ds_h2.sa.targets) + 1)]
    ds = vstack([ds_h1, ds_h2])
    ds.a.update(dss[0].a)

    randomizer = np.random.RandomState(seed = 210717)

    from mvpa2.measures.searchlight import sphere_searchlight
    from mvpa2.measures.rsa import PDistTargetSimilarity
    from mvpa2.base.learner import ChainLearner
    from mvpa2.mappers.shape import TransposeMapper
   
    # set number of permutations
    permutation_count = 2
   
    # loop over permutation count
    for i in range(permutation_count):
       
        
        # create 10 splits in the data according to a balanced leave-2-out scheme (meaning the two maps left out will always be one "up" and one "down" map)
        #splt = FactorialPartitioner(NFoldPartitioner(cvtype = 2, attr = 'chunks', selection_strategy = 'random'), attr = 'targets', selection_strategy = 'random', count = 10)
        splt = NFoldPartitioner(cvtype = 3, attr = 'chunks')
        # shuffle numbers array
        permutator = AttributePermutator('targets', limit = {'partitions': 1}, count = 1)

        # define searchlight radius, three voxels is a 12mm diameter sphere, with a volume of about 1 cubic centimeter
        radius = 3
        
        # set crossvalidation procedure to use the previously defined splits, and saves the mean accuracy value
        cv = CrossValidation(clf, splt, ChainNode([splt, permutator], space=splt.get_space()), postproc = mean_sample())
        print(ds.sa.targets)
        print(ds.sa.chunks)
        # set the searchlight procedure to use previously defined crossvalidation and radius, use a quarter of available cores (actually half, because we don't want to use the hyperthreading)
        sl = sphere_searchlight(cv, radius = radius, nproc = int(cores / 2))
        
        print('end')
