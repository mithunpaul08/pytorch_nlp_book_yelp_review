from model.classifier import ReviewClassifier
from modules.dataset import ReviewDataset
from set_parameters import Initializer
from train import Trainer



args=Initializer.set_parameters()
dataset=Initializer.read_data_make_vectorizer(args)
vectorizer = dataset.get_vectorizer()
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
Trainer.train(args)
