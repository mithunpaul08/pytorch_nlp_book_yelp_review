class Initializer():
    def set_parameters():
        args = Namespace(
            # Data and Path information
            frequency_cutoff=25,
            model_state_file='model.pth',
            review_csv='./data/yelp/reviews_with_splits_lite.csv',
            # review_csv='data/yelp/reviews_with_splits_full.csv',
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

def read_data_make_vectorizer():
    if args.reload_from_files:
        # training from a checkpoint
        print("Loading dataset and vectorizer")
        dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv,
                                                                args.vectorizer_file)
    else:
        print("Loading dataset and creating vectorizer")
        # create dataset and vectorizer
        dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
        dataset.save_vectorizer(args.vectorizer_file)
    return dataset