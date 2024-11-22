from sklearn import svm
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time
from models.svm_model import SVMModel
from utils.voting import *
from utils.evaluation_AAMI import *


class SVMDriver:
    '''
    SVM
    '''
    def __init__(self, C_value, gamma_value, multi_mode, voting_strategy, save_path, kernel='rbf', use_probability=False):
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.multi_mode = multi_mode
        self.voting_strategy = voting_strategy
        self.C_value = C_value
        self.gamma_value = gamma_value
        self.use_probability = use_probability
        self.kernel = kernel
        self.scaler = StandardScaler()

        
    def train(self, train_file):
        train_X, train_labels, train_patients = joblib.load(train_file)
        class_weights = {}
        for c in range(4):
            class_weights.update({c:len(train_labels) / float(np.count_nonzero(train_labels == c))})
        if self.gamma_value != 0.0:
            self.model = SVMModel(self.C_value, gamma_value=self.gamma_value, use_probability=self.use_probability, multi_mode=self.multi_mode, kernel=self.kernel, class_weights=class_weights)
        else:
            self.model = SVMModel(self.C_value, gamma_value='auto', use_probability=self.use_probability, multi_mode=self.multi_mode, kernel=self.kernel, class_weights=class_weights)
        
        self.model_path = os.path.join(self.save_path, 'model.pkl')
        print(f'model saved in {self.model_path}')
        # self.scaler.fit(train_X)
        # train_X_scaled = self.scaler.transform(train_X)
        # Let's Train!
        start = time.time()
        self.model.fit(train_X, train_labels) 
        end = time.time()
        print("Trained completed!\n\t" + self.model_path + "\n \
                \tTime required: " + str(format(end - start, '.2f')) + " sec" )
        # Export model: save/write trained SVM model
        joblib.dump(self.model, self.model_path)

    def write_result(self, decision_ovo, test_labels):
        if self.voting_strategy == 'ovo_voting':
            predict_ovo, counter = ovo_voting(decision_ovo, 4)

        elif self.voting_strategy == 'ovo_voting_both':
            predict_ovo, counter    = ovo_voting_both(decision_ovo, 4)

        elif self.voting_strategy == 'ovo_voting_exp':
            predict_ovo, counter    = ovo_voting_exp(decision_ovo, 4)

        # svm_model.predict_log_proba  svm_model.predict_proba   svm_model.predict ...
        perf_measures = compute_AAMI_performance_measures(predict_ovo, test_labels)

        # Write results and also predictions on 2
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if self.gamma_value != 0.0:
            write_AAMI_results(perf_measures, self.save_path + '/' + 'C_' + str(self.C_value) + 'g_' + str(self.gamma_value) + 
                '_score_Ijk_' + str(format(perf_measures.Ijk, '.2f')) + '_' + self.voting_strategy + '.txt')
        else:
            write_AAMI_results(perf_measures, self.save_path + '/' + 'C_' + str(self.C_value) + 'g_' + 
                '_score_Ijk_' + str(format(perf_measures.Ijk, '.2f')) + '_' + self.voting_strategy + '.txt')

        # Array to .csv
        if self.multi_mode == 'ovo':
            if self.gamma_value != 0.0:
                np.savetxt(self.save_path + '/' + 'C_' + str(self.C_value) + 'g_' + str(self.gamma_value) + 
                    '_decision_ovo.csv', decision_ovo)
                np.savetxt(self.save_path + '/' + 'C_' + str(self.C_value) + 'g_' + str(self.gamma_value) + 
                    '_predict_' + self.voting_strategy + '.csv', predict_ovo.astype(int), '%.0f') 
            else:
                np.savetxt(self.save_path + '/' + 'C_' + str(self.C_value) +
                    '_decision_ovo.csv', decision_ovo)
                np.savetxt(self.save_path + '/' + 'C_' + str(self.C_value) + 
                    '_predict_' + self.voting_strategy + '.csv', predict_ovo.astype(int), '%.0f') 

        elif self.multi_mode == 'ovr':
            np.savetxt(self.save_path + '/' + 'C_' + str(self.C_value) +
                '_decision_ovr.csv', decision_ovo)
            np.savetxt(self.save_path + '/' + 'C_' + str(self.C_value) + 
                '_predict_' + self.voting_strategy + '.csv', predict_ovo.astype(int), '%.0f') 

        print("Results writed at " + self.save_path + '/' + 'C_' + str(self.C_value))
    
    def test(self, test_file):
        test_X, test_labels, test_patients = joblib.load(test_file)
        # test_X_scaled = self.scaler.transform(test_X)
        print("Testing model on 2: " + self.model_path + "...")
        preds = self.model.predict(test_X) 
        pred_probs = self.model.predict_prob(test_X) # (N, class_nums)
        acc = save_accuracy_result(preds, test_labels, save_path=self.save_path)
        save_predictions(preds, test_labels, pred_probs, self.save_path)
        
        if self.multi_mode == 'ovo':
            decision_ovo = self.model.decision_function(test_X)
        self.write_result(decision_ovo, test_labels)
        return acc