import pickle
import argparse
import inspect
import os
import sys
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import \
    TabularNeuralNetTorchModel
from network_pruning_algorithms import locally, globally
from utils.common import get_logger

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def main_feedback(args):
    args.file = os.path.realpath(args.file)
    automl_search_dir = os.path.realpath(os.path.join(
        args.file, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir))
    predictor = TabularPredictor.load(automl_search_dir,
                                      verbosity=0, require_version_match=True)
    assert isinstance(predictor, TabularPredictor)

    model_name = os.path.join(
        os.path.basename(os.path.realpath(os.path.join(args.file, os.pardir, os.pardir,
                                                       os.pardir, os.pardir, os.pardir, os.pardir))),
        os.path.basename(os.path.realpath(os.path.join(args.file, os.pardir, os.pardir, os.pardir,
                                                       os.pardir, os.pardir))))

    print(model_name)
    original_model: TabularNeuralNetTorchModel = predictor._trainer.load_model(model_name)

    algorithm = os.path.basename(args.file).replace("leaderboard_", "").replace(".csv", "")
    print(algorithm)

    with open(os.path.join(os.path.join(args.file, os.pardir, os.pardir, os.pardir, os.pardir,
                                        os.pardir, "prune_search", algorithm, f"{args.id}.pkl")), "rb") as fp:
        pruned = pickle.load(fp)
    print(pruned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(dest='action', help="Main action")

    tester = subparsers.add_parser('feedback', help='Actions for testing')
    tester.add_argument(
        'file', help='path to a result of feature_subset_stochastic_search', type=str)
    tester.add_argument(
        'id', help='id of the model to consider', type=int)
    tester.add_argument(
        '-c', '--cpu', help='number of cpu cores to assign', type=int, default=round(0.7 * os.cpu_count()))

    args = parser.parse_args()
    getattr(sys.modules[__name__], f"main_{args.action}")(args)
