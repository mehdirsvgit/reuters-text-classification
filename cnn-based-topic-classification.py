from reuters21578 import Reuters
from train import *

data_set = Reuters()
data_set.load_data('offline')
# topics =['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']:

topic = 'acq'
train_positives, train_negatives = data_set.get_text(topic, 'TRAIN')
test_positives, test_negatives = data_set.get_text(topic, 'TEST')
main(train_positives, train_negatives, test_positives, test_negatives, topic)
