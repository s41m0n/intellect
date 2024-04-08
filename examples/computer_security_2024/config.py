"""
File containing used parameters throughout the notebook execution
"""
from intellect.dataset import (ContinuouLearningAlgorithm, FeatureAvailability,
                               InputForLearn, portions_from_data)
from intellect.inspect import set_seed
from intellect.model.sklearn.pruning import \
    globally_unstructured_connections_l1
from intellect.ranking import rank_metric_permutation_sklearn

#################################
# Parameters for dataset creation
#################################

# path to datasets 2017 and 2019

DATASET_2017 = '../../datasets/CICIDS2017/'
DATASET_2019 = '../../datasets/CICIDS2019/'

# column marked as label in these datasets
LABEL = 'Label'

# labels to be considered as benign (class=0 in binary classification)
BENIGN_LABELS = ['BENIGN']

# columns to remove from datasets (session identifiers and non-commond features)
EXCLUDED_COLUMNS = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP',
                    'Destination Port', 'Protocol', 'Timestamp', 'SimillarHTTP', 'Inbound']

DATASET_OUTPUT = './dataset.h5'
DATASET_SHRINKED_OUTPUT = './dataset_shrinked.h5'

####################################
# Parameters for traffic correlation
####################################

CORRELATION_OUTPUT_DIR = 'correlations_output/'

###############################
# Parameters for model training
###############################

# model parameters
N_HIDDEN_LAYERS = 8
N_HIDDEN_UNITS = 32
DROPOUT_HIDDEN = 0.
LEARNING_RATE = 'adaptive'
ACTIVATION = 'tanh'

# training parameters
BATCH_SIZE = 512
MAX_EPOCHS = 1000
EPOCHS_WO_IMPROVE = 20

# dataset portions: Train, Validation, Finetune, Refit, Test
DATASET_PORTIONS = (0.65, 0.15, 0.2)

TRAINING_OUTPUT_DIR = 'train_output/'
TRAINING_OUTPUT_MODEL = TRAINING_OUTPUT_DIR + 'oracle'

##################################################
# Parameters for feature ranking and model pruning
##################################################

# traffic categories that only the specific client (organization) has
CLIENT_CATEGORIES = ['BENIGN', 'DDoS']

# timeout for algorithms (seconds)
TIME_LIMIT = 60*10

# sizes of the feature subsets to search
TARGET_SUBSET_RATIOS = (0.1, 0.3, 0.5, 0.8)

# ratios of connections to be pruned from the network
PRUNE_RATIOS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# explored solution for each pruning/subset ratio
EXPLORED_PER_RATIO = 100

# maximum performance drop ratio accepted
PERFORMANCE_DROP_ACCEPTED_RATIO = 0.35

RANK_PRUNE_OUTPUT_DIR = 'rank_prune_output/'

PRUNE_METHOD = globally_unstructured_connections_l1

RANK_METHOD = rank_metric_permutation_sklearn

###################################
# Parameters for client model refit
###################################

# common hyperparameters
COMMON_PARAMETERS = {
    'max_epochs': 100,
    'epochs_wo_improve': 100,
    'batch_size': BATCH_SIZE}

# all possible tested scenarios:
# o_to_o:  oracle to oracle scenario, the student/client model is a copy of the oracle model.
# o_to_po: oracle to pruned oracle scenario, where the student/client model is a pruned version of the oracle.
# o_to_eo: oracle to edge oracle scenario, where the student/client is a copy of the oracle model, but it is provided
#               with only a limited set of features
# o_to_ec: oracle to edge client scenario, where the student/client is a pruned version of the oracle model AND it is provided
#               with only a limited set of features
SCENARIOS = {
    'o_to_o': [
        {'algorithm': ContinuouLearningAlgorithm.ground_truth, 'availability': FeatureAvailability.bilateral, 'learn_input': InputForLearn.client},
        {'algorithm': ContinuouLearningAlgorithm.ground_inferred, 'availability': FeatureAvailability.bilateral, 'learn_input': InputForLearn.client},
        {'algorithm': ContinuouLearningAlgorithm.knowledge_distillation, 'availability': FeatureAvailability.bilateral, 'learn_input': InputForLearn.client, 'learn_kwargs': {'alpha': 0.5}},
        {'algorithm': ContinuouLearningAlgorithm.knowledge_distillation, 'availability': FeatureAvailability.bilateral, 'learn_input': InputForLearn.client, 'learn_kwargs': {'alpha': 1}}],
    'o_to_eo': [
        {'algorithm': ContinuouLearningAlgorithm.ground_truth, 'availability': FeatureAvailability.none, 'learn_input': InputForLearn.client},
        {'algorithm': ContinuouLearningAlgorithm.ground_inferred, 'availability': FeatureAvailability.none, 'learn_input': InputForLearn.client},
        {'algorithm': ContinuouLearningAlgorithm.knowledge_distillation, 'availability': FeatureAvailability.none, 'learn_input': InputForLearn.client, 'learn_kwargs': {'alpha': 0.5}},
        {'algorithm': ContinuouLearningAlgorithm.knowledge_distillation, 'availability': FeatureAvailability.none, 'learn_input': InputForLearn.client, 'learn_kwargs': {'alpha': 1}},

        {'algorithm': ContinuouLearningAlgorithm.ground_truth, 'availability': FeatureAvailability.oracle, 'learn_input': InputForLearn.oracle},
        {'algorithm': ContinuouLearningAlgorithm.ground_inferred, 'availability': FeatureAvailability.oracle, 'learn_input': InputForLearn.client, },
        {'algorithm': ContinuouLearningAlgorithm.knowledge_distillation, 'availability': FeatureAvailability.oracle, 'learn_input': InputForLearn.client, 'learn_kwargs': {'alpha': 0.5}},
        {'algorithm': ContinuouLearningAlgorithm.knowledge_distillation, 'availability': FeatureAvailability.oracle, 'learn_input': InputForLearn.client, 'learn_kwargs': {'alpha': 1}},

        {'algorithm': ContinuouLearningAlgorithm.ground_inferred, 'availability': FeatureAvailability.oracle, 'learn_input': InputForLearn.oracle},
        {'algorithm': ContinuouLearningAlgorithm.knowledge_distillation, 'availability': FeatureAvailability.oracle, 'learn_input': InputForLearn.oracle, 'learn_kwargs': {'alpha': 0.5}},
        {'algorithm': ContinuouLearningAlgorithm.knowledge_distillation, 'availability': FeatureAvailability.oracle, 'learn_input': InputForLearn.oracle, 'learn_kwargs': {'alpha': 1}},

        {'algorithm': ContinuouLearningAlgorithm.ground_truth, 'availability': FeatureAvailability.oracle, 'learn_input': InputForLearn.mixed},
        {'algorithm': ContinuouLearningAlgorithm.ground_inferred, 'availability': FeatureAvailability.oracle, 'learn_input': InputForLearn.mixed},
        {'algorithm': ContinuouLearningAlgorithm.knowledge_distillation, 'availability': FeatureAvailability.oracle, 'learn_input': InputForLearn.mixed, 'learn_kwargs': {'alpha': 0.5}},
        {'algorithm': ContinuouLearningAlgorithm.knowledge_distillation, 'availability': FeatureAvailability.oracle, 'learn_input': InputForLearn.mixed, 'learn_kwargs': {'alpha': 1}},
    ]}

[c.update(COMMON_PARAMETERS) for v in SCENARIOS.values() for c in v]

SCENARIOS['o_to_po'] = SCENARIOS['o_to_o']
SCENARIOS['o_to_ec'] = SCENARIOS['o_to_eo']

CLIENT_REFIT_OUTPUT_DIR = 'refit_output/'

def check_consistency(path: str):
    with open(__name__ + '.py', 'r') as fp:
        content = fp.read()
    with open(path, 'r') as fp:
        content2 = fp.read()
    if hash(content) != hash(content2):
        raise RuntimeWarning(f'Current file changed with respect to the configuration provided {path}')

def get_dataset(shrinked=True):
    set_seed()
    train, validation, test = portions_from_data(DATASET_SHRINKED_OUTPUT if shrinked else DATASET_OUTPUT,
                                                 normalize=True, benign_labels=BENIGN_LABELS, ratios=DATASET_PORTIONS)
    return train, validation, test
