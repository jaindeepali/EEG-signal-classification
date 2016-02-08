import numpy as np
from scipy import signal
import glob

targets = [
  'rest',
  'left_fist',
  'left_fist_imagine',
  'right_fist',
  'right_fist_imagine',
  'both_fist',
  'both_fist_imagine',
  'both_feet',
  'both_feet_imagine'
]


def load_data():
  data_directory = 'data'
  user = 'S008'
  file_list = glob.glob(data_directory + '/' + user + '/*.edf')

  train = []
  y_train = []
  test = []
  y_test = []

  for datafile in file_list:
    d = eegtools.io.load_edf(datafile)
    run_code = datafile[-6:-4]
    if run_code in ['01', '02', '05', '06', '09', '10', '13', '14']:
      continue

    # create band-pass filter for the  8--30 Hz where the power change is expected
    (b, a) = signal.butter(3, np.array([8, 30]) / (d.sample_rate / 2), 'band')

    # band-pass filter the EEG
    filt_data = signal.lfilter(b, a, d.X, 1)

    # extract trials
    start = []
    for i in d.annotations:
      if i[2][0] in ['T1', 'T2']:
        start.append(float(i[0]) * d.sample_rate)
    # start = [float(i[0]) * d.sample_rate for i in d.annotations]
    duration = [float(i[1]) * d.sample_rate for i in d.annotations]
    offset = [0, np.min(duration)]
    # print offset
    # print len(start)
    trials, st = eegtools.featex.windows(start, offset, filt_data)
    n = len(st)

    # extract classes
    labels = [i[2][0] for i in d.annotations]
    y = []

    for label in labels:
      if run_code in ['03','07', '11']:
        # if label == 'T0':
        #   y.append(0)
        if label == 'T1':
          y.append(1)
        if label == 'T2':
          y.append(2)
      
      if run_code in ['04','08', '12']:
        # if label == 'T0':
        #   y.append(0)
        if label == 'T1':
          y.append(1)
        if label == 'T2':
          y.append(2)
    
    y = y[0:n]
    if run_code in ['03', '04']:
      test.extend(trials)
      y_test.extend(y)
    elif run_code in ['07', '08', '11', '12']:
      train.extend(trials)
      y_train.extend(y)

  train = np.asarray(train)
  test = np.asarray(test)
  y_train = np.asarray(y_train)
  y_test = np.asarray(y_test)

  return (train, y_train, test, y_test)
