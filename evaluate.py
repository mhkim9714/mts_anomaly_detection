import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

from reconstruct import test_reconstruct
from forecast import test_transformer


## Test anomaly scoring
class AnomalyScoring:
    def __init__(self, mean, std):
        if len(std.shape) == 0:
            std = np.expand_dims(std, axis=(0,1))
        assert mean.shape[0] == std.shape[0] and mean.shape[0] == std.shape[1]
        self.mean = mean
        self.std = std

    def __call__(self, recon_error): # (32,11)
        scores = []
        for error in recon_error: # (11,)
            score = error - self.mean # (11,)
            anomaly_score = np.matmul(np.matmul(score, self.std), score.T)
            scores.append(anomaly_score)
        return np.array(scores)
    

def point_adjustment(score, label, threshold):
    predict = score > threshold
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0

    for i in range(len(predict)):
        if (actual[max(i, 0) : i + 1]).any() and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j].any():
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
        elif not actual[i].any():
            anomaly_state = False
        if anomaly_state:
            predict[i] = True

    return predict


def get_evaluation(label, anomaly_score, threshold):
    pred_PA = point_adjustment(anomaly_score, label, threshold)
    precision = precision_score(label, pred_PA, zero_division=0)
    recall = recall_score(label, pred_PA, zero_division=0)
    f1 = f1_score(label, pred_PA, zero_division=0)

    auroc = roc_auc_score(label, anomaly_score)
    return precision, recall, f1, auroc


def get_fbscore(precision, recall, beta=2):
    if (precision==0) & (recall==0):
        return 0
    fb = (1+beta**2)*precision*recall / (beta**2*precision + recall)
    return fb


def evaluation(args, cfg, best_model, test_loader, AS_calculator):
    if cfg['type'] == 'reconstruct':
        test_anomaly_score, test_label = test_reconstruct(args, best_model, test_loader, AS_calculator)
    elif cfg['type'] == 'forecast':
        test_anomaly_score, test_label = test_transformer(args, best_model, test_loader, AS_calculator)

    best_result = [0,0,0,0,0,0] # threshold, pre, rec, f1, auroc, fb
    for threshold in np.linspace(np.min(test_anomaly_score), np.max(test_anomaly_score), 100):
        precision, recall, f1, auroc = get_evaluation(test_label, test_anomaly_score, threshold)
        fb = get_fbscore(precision, recall)
        
        if f1 > best_result[3]:
            best_result = [threshold, precision, recall, f1, auroc, fb]
    
    return best_result

