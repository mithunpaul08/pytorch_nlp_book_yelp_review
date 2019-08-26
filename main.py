from model.classifier import ReviewClassifier
from modules.dataset import ReviewDataset
from set_parameters import Initializer
from train import Trainer

rte=Initializer()

args=rte.set_parameters()
dataset=rte.read_data_make_vectorizer(args)
vectorizer = dataset.get_vectorizer()
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
train_rte=Trainer()
train_rte.train(args)
