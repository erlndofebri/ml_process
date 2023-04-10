#!/usr/bin/env python
# coding: utf-8

# In[69]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.dummy import DummyClassifier

from datetime import datetime
from tqdm import tqdm
import yaml
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import hashlib


# # Load Dataset

# In[70]:


x_train = joblib.load("data/raw/x_train.pkl")
y_train = joblib.load("data/raw/y_train.pkl")

x_rus = joblib.load("data/processed/x_rus.pkl")
y_rus = joblib.load("data/processed/y_rus.pkl")

x_ros = joblib.load("data/processed/x_ros.pkl")
y_ros = joblib.load("data/processed/y_ros.pkl")

x_sm = joblib.load("data/processed/x_sm.pkl")
y_sm = joblib.load("data/processed/y_sm.pkl")

x_valid = joblib.load("data/processed/x_valid_feng.pkl")
y_valid = joblib.load("data/processed/y_valid_feng.pkl")

x_test = joblib.load("data/processed/x_test_feng.pkl")
y_test = joblib.load("data/processed/y_test_feng.pkl")


# # Log Template

# In[71]:


def time_stamp():
    return datetime.now()


# In[72]:


def create_log_template():
    logger = {
        "model_name" : [],
        "model_uid" : [],
        "training_time" : [],
        "training_date" : [],
        "performance" : [],
        "f1_score_avg" : [],
        "data_configurations" : [],
    }

    return logger


# In[73]:


def training_log_updater(current_log, log_path):
    current_log = current_log.copy()

    try:
        with open(log_path, "r") as file: #read
            last_log = json.load(file)
        file.close()
    except FileNotFoundError as ffe:
        with open(log_path, "w") as file: #write
            file.write("[]")
        file.close()
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    
    last_log.append(current_log)

    with open(log_path, "w") as file:
        json.dump(last_log, file)
        file.close()

    return last_log


# ## Baseline Model

# ### Label Encoding

# In[74]:


le_deposit = joblib.load("model/le_deposit.pkl")


# In[75]:


y_train = le_deposit.transform(y_train)
y_train


# In[76]:


# create dummy classifier
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)
# train a model
dummy_clf.fit(x_train, y_train)
# get accuracy score
baseline_score = dummy_clf.score(x_test, y_test)
baseline_score


# # Training & Evaluation

# ## Create Model Object 

# In[77]:


def training_log_updater(current_log, log_path):
    current_log = current_log.copy()

    try:
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    except FileNotFoundError as ffe:
        with open(log_path, "w") as file:
            file.write("[]")
        file.close()
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    
    last_log.append(current_log)

    with open(log_path, "w") as file:
        json.dump(last_log, file)
        file.close()

    return last_log


# In[78]:


lgr_baseline = LogisticRegression()
dct_baseline = DecisionTreeClassifier()
rfc_baseline = RandomForestClassifier()
knn_baseline = KNeighborsClassifier()
xgb_baseline = XGBClassifier()


# In[79]:


list_of_model = {
    "undersampling" : [
        { "model_name": lgr_baseline.__class__.__name__, "model_object": lgr_baseline, "model_uid": ""},
        { "model_name": dct_baseline.__class__.__name__, "model_object": dct_baseline, "model_uid": ""},
        { "model_name": rfc_baseline.__class__.__name__, "model_object": rfc_baseline, "model_uid": ""},
        { "model_name": knn_baseline.__class__.__name__, "model_object": knn_baseline, "model_uid": ""},
        { "model_name": xgb_baseline.__class__.__name__, "model_object": xgb_baseline, "model_uid": ""}
        ],
    "oversampling" : [
        { "model_name": lgr_baseline.__class__.__name__, "model_object": lgr_baseline, "model_uid": ""},
        { "model_name": dct_baseline.__class__.__name__, "model_object": dct_baseline, "model_uid": ""},
        { "model_name": rfc_baseline.__class__.__name__, "model_object": rfc_baseline, "model_uid": ""},
        { "model_name": knn_baseline.__class__.__name__, "model_object": knn_baseline, "model_uid": ""},
        { "model_name": xgb_baseline.__class__.__name__, "model_object": xgb_baseline, "model_uid": ""}
        ],
    "smote" : [
        { "model_name": lgr_baseline.__class__.__name__, "model_object": lgr_baseline, "model_uid": ""},
        { "model_name": dct_baseline.__class__.__name__, "model_object": dct_baseline, "model_uid": ""},
        { "model_name": rfc_baseline.__class__.__name__, "model_object": rfc_baseline, "model_uid": ""},
        { "model_name": knn_baseline.__class__.__name__, "model_object": knn_baseline, "model_uid": ""},
        { "model_name": xgb_baseline.__class__.__name__, "model_object": xgb_baseline, "model_uid": ""}
        ],
    }


# ## Training Model

# In[80]:


def train_eval_model(list_of_model, prefix_model_name, x_train, y_train, data_configuration_name, x_valid, y_valid, log_path):

    list_of_model = copy.deepcopy(list_of_model)
    logger = create_log_template()

    for model in tqdm(list_of_model):    
        model_name = prefix_model_name + "-" + model["model_name"]

        #fit
        start_time = time_stamp()
        model["model_object"].fit(x_train, y_train)
        finished_time = time_stamp()

        # convert ke second
        elapsed_time = finished_time - start_time
        elapsed_time = elapsed_time.total_seconds()
        

        # using predict x_valid (x_test masih disimpen)
        y_pred = model["model_object"].predict(x_valid)
        performance = classification_report(y_valid, y_pred, output_dict = True)

        # make uid (one way encryption)
        plain_id = str(start_time) + str(finished_time)
        chiper_id = hashlib.md5(plain_id.encode()).hexdigest()

        model["model_uid"] = chiper_id

        # append
        logger["model_name"].append(model_name)
        logger["model_uid"].append(chiper_id)
        logger["training_time"].append(elapsed_time)
        logger["training_date"].append(str(start_time))
        logger["performance"].append(performance)
        logger["f1_score_avg"].append(performance["macro avg"]["f1-score"]) # pake recall
        logger["data_configurations"].append(data_configuration_name)

    training_log = training_log_updater(logger, log_path)

    return training_log, list_of_model


# ### Baseline Model Undersampling

# In[81]:


training_log, list_of_model_rus = train_eval_model(
    list_of_model["undersampling"],
    "baseline_model",
    x_rus,
    y_rus,
    "undersampling",
    x_valid,
    y_valid,
    "log/training_log.json"
)


# In[82]:


list_of_model["undersampling"] = copy.deepcopy(list_of_model_rus)


# ### Baseline Model Oversampling

# In[83]:


training_log, list_of_model_ros = train_eval_model(
    list_of_model["oversampling"],
    "baseline_model",
    x_rus,
    y_rus,
    "oversampling",
    x_valid,
    y_valid,
    "log/training_log.json"
)


# In[84]:


list_of_model["oversampling"] = copy.deepcopy(list_of_model_ros)


# ### Baseline Model SMOTE

# In[85]:


training_log, list_of_model_sm = train_eval_model(
    list_of_model["smote"],
    "baseline_model",
    x_rus,
    y_rus,
    "smote",
    x_valid,
    y_valid,
    "log/training_log.json"
)


# In[86]:


list_of_model["smote"] = copy.deepcopy(list_of_model_sm)


# ## Best Performance

# In[87]:


def training_log_to_df(training_log):
    training_res = pd.DataFrame()

    for log in tqdm(training_log):
        training_res = pd.concat([training_res, pd.DataFrame(log)])
    
    training_res.sort_values(["f1_score_avg", "training_time"], ascending = [False, True], inplace = True)
    training_res.reset_index(inplace = True, drop = True)
    
    return training_res


# In[88]:


training_res = training_log_to_df(training_log)


# In[89]:


training_res


# In[90]:


def get_best_model(training_log_df, list_of_model):
    model_object = None

    # pick the highest one
    best_model_info = training_log_df.sort_values(["f1_score_avg", "training_time"], ascending = [False, True]).iloc[0]
    
    for configuration_data in list_of_model:
        for model_data in list_of_model[configuration_data]:
            if model_data["model_uid"] == best_model_info["model_uid"]:
                model_object = model_data["model_object"]
                break
    
    if model_object == None:
        raise RuntimeError("The best model not found in your list of model.")
    
    return model_object
    


# In[91]:


model = get_best_model(training_res, list_of_model)


# In[92]:


model


# In[93]:


joblib.dump(model, "model/model.pkl")


# ### Hyperparameter Tuning

# In[94]:


params_rf = {
    "n_estimators" : [5,7,12,21],
    "max_depth" : [2,3,6,16,17,30],
    "min_samples_split" : [4,6,22,29],
    "min_samples_leaf" : [6,7,12,28,32]
}


# In[95]:


rf_param = GridSearchCV(RandomForestClassifier(),
                        params_rf)


# In[96]:


list_of_model["undersampling"].append({"model_name": rf_param.__class__.__name__ + "-" + rf_param.estimator.__class__.__name__, "model_object": copy.deepcopy(rf_param), "model_uid": ""})
list_of_model["oversampling"].append({"model_name": rf_param.__class__.__name__ + "-" + rf_param.estimator.__class__.__name__, "model_object": copy.deepcopy(rf_param), "model_uid": ""})
list_of_model["smote"].append({"model_name": rf_param.__class__.__name__ + "-" + rf_param.estimator.__class__.__name__, "model_object": copy.deepcopy(rf_param), "model_uid": ""})


# In[97]:


training_log, list_of_model_sm_hyp = train_eval_model(
    [list_of_model["smote"][-1]], # newest
    "hyperparams",
    x_sm,
    y_sm,
    "smote",
    x_valid,
    y_valid,
    "log/training_log.json"
)


# In[99]:


list_of_model["smote"][-1] = copy.deepcopy(list_of_model_sm_hyp[0])


# In[100]:


list_of_model["smote"][-1]


# In[101]:


training_log_to_df(training_log)


# ## Confusion Matrix

# In[102]:


y_pred = model.predict(x_valid)


# In[103]:


ConfusionMatrixDisplay.from_predictions(y_valid, y_pred)


# ## Classification Report

# In[104]:


print(classification_report(y_valid, y_pred))


# In[105]:


baseline_score


# ## Evaluate with test data

# In[106]:


y_pred2 = model.predict(x_test)


# In[107]:


print(classification_report(y_test, y_pred2))


# In[ ]:




