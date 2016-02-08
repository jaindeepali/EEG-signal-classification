import numpy as np
from sklearn import svm, pipeline, base, metrics
import eegtools
import load_data from load_data

'''
Training model for classification of EEG samples into motor imagery classes
'''

# Create sklearn-compatible feature extraction and classification pipeline:
class CSP(base.BaseEstimator, base.TransformerMixin):
  def fit(self, X, y):
    class_covs = []

    # calculate per-class covariance
    for ci in np.unique(y): 
      class_covs.append(np.cov(np.hstack(X[y==ci])))

    # calculate CSP spatial filters
    self.W = eegtools.spatfilt.csp(class_covs[0], class_covs[1], 6)
    return self


  def transform(self, X):
    # Note that the projection on the spatial filter expects zero-mean data.
    return np.asarray([np.dot(self.W, trial) for trial in X])


class ChanVar(base.BaseEstimator, base.TransformerMixin):
  def fit(self, X, y): return self
  def transform(self, X):
    return np.var(X, axis=2)  # X.shape = (trials, channels, time)

def train():
  (train, y_train, test, y_test) = load_data()

  pipe = pipeline.Pipeline(
    [('csp', CSP()), ('chan_var', ChanVar()), ('svm', svm.SVC(kernel='linear'))])

  # train model
  pipe.fit(train, y_train)

  # make predictions on unseen test data
  y_pred = pipe.predict(test)

  print metrics.classification_report(y_test, y_pred)

if __name__ == '__main__':
  train()