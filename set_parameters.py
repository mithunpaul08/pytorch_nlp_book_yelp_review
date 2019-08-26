from argparse import Namespace
import torch
import os
from utils.utils import set_seed_everywhere
from utils.utils import handle_dirs
from modules.dataset import ReviewDataset

class Initializer():
    def set_parameters(self):
        args = Namespace(
            # Data and Path information
            frequency_cutoff=25,
            model_state_file='model.pth',
            fever_lex_train='data-local/rte/fever/train/fever_train_lex_3labels_200_smartner_3labels_no_lists_evidence_not_sents.jsonl',
            fever_lex_dev='data-local/rte/fever/dev/fever_dev_lex_3labels_200_no_lists_evidence_not_sents.jsonl',
            save_dir='model_storage/ch3/yelp/',
            vectorizer_file='vectorizer.json',
            # No Model hyper parameters
            # Training hyper parameters
            batch_size=128,
            early_stopping_criteria=5,
            learning_rate=0.001,
            num_epochs=2,
            seed=1337,
            # Runtime options
            catch_keyboard_interrupt=True,
            cuda=True,
            expand_filepaths_to_save_dir=True,
            reload_from_files=False,
            truncate_words_length=1000,
            type_of_data='plain'
        )

        if args.expand_filepaths_to_save_dir:
            args.vectorizer_file = os.path.join(args.save_dir,
                                                args.vectorizer_file)

            args.model_state_file = os.path.join(args.save_dir,
                                                 args.model_state_file)

            print("Expanded filepaths: ")
            print("\t{}".format(args.vectorizer_file))
            print("\t{}".format(args.model_state_file))

        # Check CUDA
        if not torch.cuda.is_available():
            args.cuda = False

        print("Using CUDA: {}".format(args.cuda))

        args.device = torch.device("cuda" if args.cuda else "cpu")

        # Set seed for reproducibility
        set_seed_everywhere(args.seed, args.cuda)

        # handle dirs
        handle_dirs(args.save_dir)

        return args

    def read_data_make_vectorizer(self,args):
        if args.reload_from_files:
            # training from a checkpoint
            print("Loading dataset and vectorizer")
            dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv,
                                                                     args.vectorizer_file)
        else:
            print("Loading dataset and creating vectorizer")
            # create dataset and vectorizer
            dataset = ReviewDataset.load_dataset_and_make_vectorizer(args)
            dataset.save_vectorizer(args.vectorizer_file)
        return dataset