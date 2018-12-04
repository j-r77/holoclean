import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import InitFeaturizer
from repair.featurize import InitAttFeaturizer
from repair.featurize import InitSimFeaturizer
from repair.featurize import FreqFeaturizer
from repair.featurize import OccurFeaturizer
from repair.featurize import ConstraintFeat
from repair.featurize import LangModelFeat

hc = holoclean.HoloClean(pruning_topk=0.1, epochs=30, weight_decay=0.01, threads=20, batch_size=1, verbose=True, timeout=3*60000).session

#hc.load_data('adult', '.', 'Adult20.csv')
#REPAIRS
#(11, u'relationship', 'Husband', u'Husband')
#(11, u'sex', 'Female', u'Male')
#(4, u'relationship', 'Husband', u'Husband')
#(4, u'sex', 'Female', u'Male')

#hc.load_data('adult', '.', 'Adult1100.csv')

#REPAIRS
#(4, u'relationship', 'Husband', u'Husband')
#(4, u'sex', 'Female', u'Female')
#(11, u'relationship', 'Husband', u'Husband')
#(11, u'sex', 'Female', u'Female')
#(652, u'relationship', 'Husband', u'Husband')
#(652, u'sex', 'Female', u'Female')
#(689, u'relationship', 'Wife', u'Wife')
#(689, u'sex', 'Male', u'Male')
#(504, u'relationship', 'Husband', u'Husband')
#(504, u'sex', 'Female', u'Female')
#(26, u'relationship', 'Husband', u'Husband')
#(26, u'sex', 'Female', u'Female')
#(1084, u'relationship', 'Husband', u'Husband')
#(1084, u'sex', 'Female', u'Female')
#(703, u'relationship', 'Wife', u'Wife')
#(703, u'sex', 'Male', u'Male')

hc.load_data('adult', '.', 'AdultFull.csv')
#REPAIRS
#(8736, u'relationship', 'Wife', u'Wife')
#(8736, u'sex', 'Male', u'Male')
#(4, u'relationship', 'Husband', u'Husband')
#(4, u'sex', 'Female', u'Female')
#(11, u'relationship', 'Husband', u'Husband')
#(11, u'sex', 'Female', u'Female')
#(652, u'relationship', 'Husband', u'Husband')
#(652, u'sex', 'Female', u'Female')
#(689, u'relationship', 'Wife', u'Wife')
#(689, u'sex', 'Male', u'Male')
#(504, u'relationship', 'Husband', u'Husband')
#(504, u'sex', 'Female', u'Female')
#(3929, u'relationship', 'Husband', u'Husband')
#(3929, u'sex', 'Female', u'Female')
#(26, u'relationship', 'Husband', u'Husband')
#(26, u'sex', 'Female', u'Female')
#(1084, u'relationship', 'Husband', u'Husband')
#(1084, u'sex', 'Female', u'Female')
#(16666, u'relationship', 'Wife', u'Wife')
#(16666, u'sex', 'Male', u'Male')
#(14046, u'relationship', 'Wife', u'Wife')
#(14046, u'sex', 'Male', u'Male')
#(703, u'relationship', 'Wife', u'Wife')
#(703, u'sex', 'Male', u'Male')

hc.load_dcs('.', 'adult_fbis.txt')
#DCS
#t1&EQ(t1.Sex,"Female")&EQ(t1.Relationship,"Husband")
#t1&EQ(t1.Sex,"Male")&EQ(t1.Relationship,"Wife")
hc.ds.set_constraints(hc.get_dcs())
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)

# 4. Repair errors utilizing the defined features.
hc.setup_domain()
featurizers = [InitAttFeaturizer(), InitSimFeaturizer(), FreqFeaturizer(), OccurFeaturizer()]#, LangModelFeat(), ConstraintFeat()]
hc.repair_errors(featurizers)