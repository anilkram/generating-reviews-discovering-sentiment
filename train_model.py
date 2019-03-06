from encoder import Model
from matplotlib import pyplot as plt
from utils import sst_binary, train_with_reg_cv
import pickle

model = Model()

trX, vaX, teX, trY, vaY, teY = sst_binary()
trX = trX + teX
trY = trY + teY
print('Done loading dataset')
print('Train set size: {0} and Validation set size: {1}'.format(len(trX), len(vaX)))
trXt = model.transform(trX)
vaXt = model.transform(vaX)
print('Done extracting embeddings for dataset')
print(trXt.shape)
print(vaXt.shape)

# classification results
full_rep_acc, c, nnotzero, model = train_with_reg_cv(trXt, trY, vaXt, vaY)
print('%05.2f test accuracy'%full_rep_acc)
print('%05.2f regularization coef'%c)
print('%05d features used'%nnotzero)
model_path = 'model/logreg'
print('Model saved at {}'.format(model_path))
