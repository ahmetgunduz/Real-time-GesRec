import csv
import pdb
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

class Queue:
    #Constructor creates a list
    def __init__(self, max_size, n_classes):
        self.queue = list(np.zeros((max_size, n_classes),dtype = float).tolist())
        self.max_size = max_size
        self.median = None
        self.ma = None
        self.ewma = None
    #Adding elements to queue
    def enqueue(self,data):
        self.queue.insert(0,data)
        self.median = self._median()
        self.ma = self._ma()
        self.ewma = self._ewma()
        return True

    #Removing the last element from the queue
    def dequeue(self):
        if len(self.queue)>0:
            return self.queue.pop()
        return ("Queue Empty!")

    #Getting the size of the queue
    def size(self):
        return len(self.queue)

    #printing the elements of the queue
    def printQueue(self):
        return self.queue

    #Average   
    def _ma(self):
        return np.array(self.queue[:self.max_size]).mean(axis = 0)

    #Median
    def _median(self):
        return np.median(np.array(self.queue[:self.max_size]), axis = 0)
    
    #Exponential average
    def _ewma(self):
        weights = np.exp(np.linspace(-1., 0., self.max_size))
        weights /= weights.sum()
        average = weights.reshape(1,self.max_size).dot( np.array(self.queue[:self.max_size]))
        return average.reshape(average.shape[1],)

# def LevenshteinDistance(a,b):
#     # This is a straightforward implementation of a well-known algorithm, and thus
#     # probably shouldn't be covered by copyright to begin with. But in case it is,
#     # the author (Magnus Lie Hetland) has, to the extent possible under law,
#     # dedicated all copyright and related and neighboring rights to this software
#     # to the public domain worldwide, by distributing it under the CC0 license,
#     # version 1.0. This software is distributed without any warranty. For more
#     # information, see <http://creativecommons.org/publicdomain/zero/1.0>
#     "Calculates the Levenshtein distance between a and b."
#     n, m = len(a), len(b)
#     if n > m:
#         # Make sure n <= m, to use O(min(n,m)) space
#         a,b = b,a
#         n,m = m,n
        
#     current = range(n+1)
#     for i in range(1,m+1):
#         previous, current = current, [i]+[0]*n
#         for j in range(1,n+1):
#             add, delete = previous[j]+1, current[j-1]+1
#             change = previous[j-1]
#             if a[j-1] != b[i-1]:
#                 change = change + 1
#             current[j] = min(add, delete, change)
            
#     return current[n]


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def calculate_precision(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return  precision_score(targets.cpu().view(-1), pred.cpu().view(-1), average = 'macro')


def calculate_recall(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return  recall_score(targets.cpu().view(-1), pred.cpu().view(-1), average = 'macro')


#############################################
######## For Levenshtein Accuracy calculation

def LevenshteinDistance(r, h):
    edit_distance_matrix = editDistance(r=r, h=h)
    step_list = getStepList(r=r, h=h, d=edit_distance_matrix)

    min_distance = float(edit_distance_matrix[len(r)][len(h)])
    num_del = float(np.sum([s == "d" for s in step_list]))
    num_ins = float(np.sum([s == "i" for s in step_list]))
    num_sub = float(np.sum([s == "s" for s in step_list]))

    word_error_rate = round((min_distance / len(r) * 100), 4)
    del_rate = round((num_del / len(r) * 100), 4)
    ins_rate = round((num_ins / len(r) * 100), 4)
    sub_rate = round((num_sub / len(r) * 100), 4)

    # return {"wer": word_error_rate, "del": del_rate, "ins": ins_rate, "sub": sub_rate}
    return min_distance, num_del, num_ins, num_sub

def editDistance(r, h):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
        (len(r) + 1, len(h) + 1)
    )
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def getStepList(r, h, d):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    """
    x = len(r)
    y = len(h)
    max_len = 3 * (x + y)
    list = []
    while True:
        if (x <= 0 and y <= 0) or (len(list) > max_len):
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            list.append("e")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif y >= 1 and d[x][y] == d[x][y - 1] + 1:
            list.append("i")
            x = max(x, 0)
            y = max(y - 1, 0)
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + 1:
            list.append("s")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        else:
            list.append("d")
            x = max(x - 1, 0)
            y = max(y, 0)
    return list[::-1]