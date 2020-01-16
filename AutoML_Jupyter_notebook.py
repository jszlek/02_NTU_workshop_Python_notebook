import os
import string
import random
import glob
import h2o
import statistics
import psutil
import csv
import sys
import re
import shap
import pickle
import pandas as pd
import numpy as np
from h2o.automl import H2OAutoML
from pathlib import Path
from random import randint
from math import sqrt
from sklearn.model_selection import GroupShuffleSplit
from copy import copy
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GroupKFold # import KFold
from sklearn import preprocessing
# %matplotlib inline - this line must be commented-out in order to run in a Python console
import matplotlib.pyplot as plt

# -------------------------------
# Random key generator - function
# -------------------------------
def random_key_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
# -------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# H2OPredWrapper - class used for obtaining predictions understood by shap package
# --------------------------------------------------------------------------------
class H2OPredWrapper:
    def __init__(self, h2o_model, feature_names):
        self.h2o_model = h2o_model
        self.feature_names = feature_names
    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        self.dataframe= pd.DataFrame(X, columns=self.feature_names)
        self.predictions = self.h2o_model.predict(h2o.H2OFrame(self.dataframe)).as_data_frame().values
        return self.predictions.astype('float64')[:,-1]
# --------------------------------------------------------------------------------

server_multicore: bool = True
my_max_ram_allowed: int = 16

# user specified seeds
my_seed_FS: int = 1
my_seed_FS_10cv: int = 1

# -------------------------
# General options:
# -------------------------
fs_data: str = 'PLGA_300in_SR_BAZA.txt' # Please provide full filename if perform_FS is True or classic_approach without n-fold cv are to be run

my_keep_cross_validation_predictions: bool = True
my_keep_cross_validation_models: bool = True
my_keep_cross_validation_fold_assignment: bool = True

# save pojo or mojo model boolean = True/False
save_pojo_or_mojo: bool = True

# Which column contains indicies to make split - 1 = 1st col, 2 = 2nd col etc. 
index_column: int = 1

# --------------------------
# Feature selection options:
# --------------------------
# Feature selection AutoML execution time
FS_h2o_max_runtime_secs: int = 45
FS_h2o_max_runtime_secs_2nd_time: int = 10*60

# How many loops of FS
my_FS_loops: int = 6

# Scale by original score or RMSE - this is only for comparison with fscaret - set False to scale by the RMSE
# - only if perform_FS is True
original_scale: bool = True

# create core file name - used only when perform_FS is True to create filenames
core_filename: str = 'new_PLGA'

# Manually include features, eg. 'Time_min' in dissolution profile 
# - used only when perform_FS is True
include_features: list = []

# Feature selection threshold - range = [0; 1] - usually between 0.01 and 0.001
# - only if perform_FS is True
fs_threshold: float = 0.01

# Feature selection short loop RMSE threshold
# - only if perform_FS is True
rmse_fs_short_loop_threshold: float = 15.0


# --------------------------
# Cross validation options:
# --------------------------

# How many fold in cross validation is used only if perform_FS is True
no_folds: int = 10

# 10cv AutoML execution time after feature selection
h2o_max_runtime_secs_10cv: int = 20*60

# generate random port number
# -------------------------------
my_port_number = random.randint(54322,65000)

# Create three random strings
# -------------------------
aml_name = 'A' + random_key_generator(15) # for FS project name
aml2_name = 'A' + random_key_generator(15) # for 10-cv project name
cluster_name = 'A' + random_key_generator(15) # for h2o cluster name

# get current directory (PosixPath)
# -----------------------
my_current_dir = Path.cwd()

# get export directory and other subdirs (PosixPath)
# -----------------------
my_export_dir = my_current_dir.joinpath(str(my_current_dir) + '/export')
my_10cv_FS_dir = my_current_dir.joinpath(str(my_current_dir) + '/10cv_FS')
my_10cv_orig_dir = my_current_dir.joinpath(str(my_current_dir) + '/10cv_orig')
my_test_external = my_current_dir.joinpath(str(my_current_dir) + '/test_external')
my_pojo_or_mojo_FS = my_current_dir.joinpath(str(my_current_dir) + '/pojo_or_mojo_FS')
my_pojo_or_mojo_10cv = my_current_dir.joinpath(str(my_current_dir) + '/pojo_or_mojo_10cv')
my_model_FS = my_current_dir.joinpath(str(my_current_dir) + '/model_FS')
my_model_10cv = my_current_dir.joinpath(str(my_current_dir) + '/model_10cv')

# check subdirectory structure
# ----------------------------------------
Path(my_export_dir).mkdir(parents=True, exist_ok=True)
Path(my_10cv_FS_dir).mkdir(parents=True, exist_ok=True)
Path(my_10cv_orig_dir).mkdir(parents=True, exist_ok=True)
Path(my_test_external).mkdir(parents=True, exist_ok=True)
Path(my_pojo_or_mojo_FS).mkdir(parents=True, exist_ok=True)
Path(my_pojo_or_mojo_10cv).mkdir(parents=True, exist_ok=True)
Path(my_model_FS).mkdir(parents=True, exist_ok=True)
Path(my_model_10cv).mkdir(parents=True, exist_ok=True)

# check runtime mode - either many servers on the machine (server_multicore = F) or one server per one machine (server_multicore = T)
# -------------------------------------------
if server_multicore is True:
    my_cores = psutil.cpu_count() - 2
else:
    my_cores = 1

# check system free mem and apply it to the server
# ------------------------------------------------
memfree = psutil.virtual_memory().total
memfree_g = int(round(memfree/1024/1024/1024,3))

if memfree_g < 2:
 memfree_g = 2+'G'

if my_max_ram_allowed > 0:
  memfree_g = str(my_max_ram_allowed)+'G'
  
# -------------------------------------
# run h2o server
# -------------------------------------
h2o.init(nthreads=my_cores, 
         min_mem_size=memfree_g,
         max_mem_size=memfree_g,
         port=my_port_number,
         ice_root=str(my_export_dir),
         name=str(cluster_name),
         start_h2o=True)
# -------------------------------------

print(h2o.cluster().list_all_extensions())

if 'AutoML' in h2o.cluster().list_all_extensions():
    print('Congrats! AutoML is available!')
if 'AutoML' not in h2o.cluster().list_all_extensions():
    print('Please check your configuration, AutoML is NOT available')

data = pd.read_csv(fs_data, sep='\t', engine='python')

data.head(15)
data.tail(5)
data.describe()

fig_data = data[['Formulation_no','Time_Days','Q_perc']]
fig_data = fig_data.groupby(['Formulation_no'])
fig, ax = plt.subplots(23,3, figsize=(12,48), sharex='all', sharey = 'all', squeeze = False)

for i in fig_data.groups:
    tmp_data = fig_data.get_group(i)
    plt.subplot(23,3,i)
    plt.subplots_adjust(hspace = 0.4)
    plt.ylim(0,100)
    plt.plot(tmp_data['Time_Days'],tmp_data['Q_perc'])
    plt.text(0, 105, str('Form no ' + str(i)), weight="bold")
    plt.legend()

# SAVE figure
plt.savefig('Q_perc_observed.pdf', format='pdf', dpi=1200)

# checking if my_10cv_FS_dir, my_10cv_orig_dir, my_pojo_or_mojo_FS, my_pojo_or_mojo_10cv, my_model_FS, my_model_10cv are empty if not delete content
print('Checking for non-empty dirs ...')
print('')
checking_list = [my_10cv_FS_dir, my_10cv_orig_dir, my_pojo_or_mojo_FS, my_pojo_or_mojo_10cv, my_model_FS, my_model_10cv, my_export_dir]
for checked_dir in checking_list:
    if len(os.listdir(checked_dir)) > 0:
        print('Removing files from ' + str(checked_dir) + ':')
        files_to_remove = glob.glob(str(checked_dir.joinpath(str(checked_dir)+'/*')))
        for f in files_to_remove:
            # print(str(f))
            if os.path.isfile(f):
                os.remove(f)
            else: # if error occurs #
                print("Error: %s file not found" %f)

ncols = data.shape[1]-1
nrows = data.shape[0]

X = data.drop(data.columns[[0,ncols]], axis=1)
y = data[data.columns[ncols]]

# needed to make cv by groups - first column contains indicies!
groups = data[data.columns[[index_column - 1]]]

# Define FS_loops counter
no_FS_loops = 1

# the counter is set from 1, therefore = my_FS_loops + 1
while no_FS_loops < (my_FS_loops + 1): 

    # split on train - test dataset by group 'Formulation no' - this is for Feature Selection
    tmp_train_inds, tmp_test_inds = next(GroupShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3).split(X, groups=groups))
    tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = X.iloc[tmp_train_inds], X.iloc[tmp_test_inds], y.iloc[tmp_train_inds], y.iloc[tmp_test_inds]

    # Write splits on disk
    tmp_train_set = pd.concat([tmp_X_train, tmp_y_train], axis=1)
    tmp_test_set = pd.concat([tmp_X_test, tmp_y_test], axis=1)

    tmp_y_idx = tmp_train_set.columns[tmp_train_set.shape[1]-1]

    tmp_training_frame = h2o.H2OFrame(tmp_train_set)
    tmp_testing_frame = h2o.H2OFrame(tmp_test_set)

    # print out no of loop
    print('\n' + 'Starting FS loop no: ' + str(no_FS_loops) + '\n')
    tmp_my_random_seed_FS = random.randint(1,100000000)
    print('Temp random seed: ' + str(tmp_my_random_seed_FS) + '\n')

    tmp_aml_name = 'A' + random_key_generator(15)

    # autoML settings
    tmp_FS_model = H2OAutoML(max_runtime_secs = FS_h2o_max_runtime_secs,
                             seed = tmp_my_random_seed_FS,
                             project_name = tmp_aml_name,
                             nfolds = 0,
                             export_checkpoints_dir = str(my_export_dir),
                             keep_cross_validation_models = my_keep_cross_validation_models,
                             keep_cross_validation_predictions = my_keep_cross_validation_predictions,
                             keep_cross_validation_fold_assignment = my_keep_cross_validation_fold_assignment,
                             verbosity = 'info',
                             sort_metric = 'RMSE')

    # train model for FS
    tmp_FS_model.train(y = tmp_y_idx, training_frame = tmp_training_frame, leaderboard_frame = tmp_testing_frame)

    # write first model rmse metrics
    if no_FS_loops is 1:
        tmp_FS_rmse = tmp_FS_model.leader.model_performance(tmp_testing_frame)['RMSE']
        aml_name = tmp_aml_name
        my_random_seed_FS = tmp_my_random_seed_FS

    # print out RMSE for the model
    print('\n' + 'RMSE for FS loop no: ' + str(no_FS_loops) + ' is ' + str(tmp_FS_model.leader.model_performance(tmp_testing_frame)['RMSE']) + '\n')

    # if new tmp_FS_model has better performance overwrite it to aml
    if tmp_FS_model.leader.model_performance(tmp_testing_frame)['RMSE'] < tmp_FS_rmse:

        # overwrite rmse for the tmp_FS_model - the leader
        tmp_FS_rmse = tmp_FS_model.leader.model_performance(tmp_testing_frame)['RMSE']

        #generate an unique file name based on the id and record
        file_name_train = str(core_filename)+"_h2o_train_for_FS"+".txt"
        file_name_test = str(core_filename)+"_h2o_test_for_FS"+".txt"

        tmp_train_set.to_csv(file_name_train, index=False, sep="\t")
        tmp_test_set.to_csv(file_name_test, index = False, sep="\t")

        y_idx = tmp_y_idx

        training_frame = tmp_training_frame
        testing_frame = tmp_testing_frame

        my_random_seed_FS = tmp_my_random_seed_FS
        aml_name = tmp_aml_name

        print('Current best aml name: ' + str(aml_name))
        print('Current best seed: ' + str(my_random_seed_FS) + '\n')

        # if new tmp_FS_model RMSE is lower or equal has better performance overwrite it to aml
        if tmp_FS_model.leader.model_performance(tmp_testing_frame)['RMSE'] <= rmse_fs_short_loop_threshold:

            print('\n' + 'Performance of obtained model is better than set threshold: ' + '\n')
            print('Threshold was set to: ' + str(rmse_fs_short_loop_threshold) + '\n')
            print('Performance of obtained model is: ' + str(tmp_FS_rmse) + '\n')
            print('Breaking the short FS loop')

            # Making no_FS_loops equal to my_FS_loops to break the while loop
            no_FS_loops = my_FS_loops 


    # FS_loop counter +1
    no_FS_loops += 1

# Once again perform FS on 'the best' train / test dataset, but this time for much longer

print('\n' + 'Used best aml name: ' + str(aml_name))
print('Used best seed: ' + str(my_random_seed_FS) + '\n')

# autoML settings
aml = H2OAutoML(max_runtime_secs = FS_h2o_max_runtime_secs_2nd_time,
                seed = my_random_seed_FS,
                project_name = aml_name,
                nfolds = 0,
                export_checkpoints_dir = str(my_export_dir),
                keep_cross_validation_models = my_keep_cross_validation_models,
                keep_cross_validation_predictions = my_keep_cross_validation_predictions,
                keep_cross_validation_fold_assignment = my_keep_cross_validation_fold_assignment,
                verbosity = 'info',
                sort_metric = 'RMSE')

# train model for FS
aml.train(y = y_idx, training_frame = training_frame, leaderboard_frame = testing_frame)

# saving model
my_model_FS_path = h2o.save_model(aml.leader, path = './model_FS')

print('')
print('Final model of feature selection is located at: ')
print(str(my_model_FS_path))
print('')

# Download POJO or MOJO
if save_pojo_or_mojo is True:
    if aml.leader.have_pojo is True:
        aml.leader.download_pojo(get_genmodel_jar = True, path = './pojo_or_mojo_FS')
    if aml.leader.have_mojo is True:
        aml.leader.download_mojo(get_genmodel_jar = True, path = './pojo_or_mojo_FS')

# get leader model key
model_key = aml.leader.key

lb = aml.leaderboard
lbdf = lb.as_data_frame()

print("\n","Leaderboard (10 best models): ")
lbdf.head(n=10)

if ("StackedEnsemble" in model_key) is True:
    model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])

    # get the best model key
    m = h2o.get_model(model_ids[0])

    # get the metalearner model
    meta = h2o.get_model(m.metalearner()['name'])

    # get varimp_df from metalearner
    if ('glm' in meta.algo) is True:
        varimp_df = pd.DataFrame.from_dict((meta.coef()), orient = 'index')
        varimp_df = varimp_df[1:] # omit Intercept
    else:
        varimp_df = pd.DataFrame(meta.varimp())

    model_list = []

    for model in m.params['base_models']['actual']:
        model_list.append(model['name'])

    print('')
    print('Models used in feature selection:')
    print(*model_list, sep='\n')
    print('')

    # create two dictionaries for storing variable importance and rmse
    var_imp_models = dict([(key, []) for key in model_list])
    rmse_df = dict([(key, []) for key in model_list])


    # get variable importance and rmse from base learners
    for model in model_list:
        tmp_model = h2o.get_model(str(model))

        # check if tmp_model has varimp()
        if tmp_model.varimp() is None:
            print(str(model))
            del var_imp_models[str(model)]
        else:
            # check if tmp_model is glm - it has no varimp() but coef()
            if ('glm' in tmp_model.algo) is True:
                tmp_var_imp = pd.DataFrame.from_dict(tmp_model.coef(), orient = 'index').rename(columns={0:'scaled_importance'})
                tmp_var_imp = tmp_var_imp[1:] # omit Intercept
                tmp_var_imp.insert(loc = 0, column = 'variable', value = tmp_var_imp.index) # reset index of rows into column
            else:
                tmp_var_imp = tmp_model.varimp(use_pandas=True).iloc[:,[0,2]]

            tmp_rmse = tmp_model.rmse()
            var_imp_models[str(model)].append(tmp_var_imp)
            rmse_df[str(model)].append(tmp_rmse)

    if original_scale is False:
        rmse_df = pd.DataFrame(rmse_df.values())
        rmse_sum = rmse_df.sum()[0]
        rmse_scale = rmse_sum / rmse_df

        x = rmse_scale.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        rmse_scale = pd.DataFrame(x_scaled)
        rmse_scale = pd.DataFrame(data=rmse_scale.values,index=model_list)

        for idx in rmse_scale.iterrows():
            var_imp_models[str(idx[0])][0]['scaled_importance'] = var_imp_models[str(idx[0])][0].values[0:,1] * idx[1].values

    elif original_scale is True:
        meta_scale = varimp_df
        for idx in meta_scale.iterrows():
            if ('glm' in meta.algo) is True:
                var_imp_models[str(idx[0])][0]['scaled_importance'] = var_imp_models[str(idx[0])][0].values[0:,1] * float(idx[1])
            else:
                var_imp_models[str(idx[1][0])][0]['scaled_importance'] = var_imp_models[str(idx[1][0])][0]['scaled_importance'] * idx[1][3]

    # new dataframe init     
    scaled_var_imp_df = pd.DataFrame()

    for idx in var_imp_models.keys():
        df_tmp = var_imp_models[str(idx)][0]['scaled_importance']
        df_tmp.index = var_imp_models[str(idx)][0]['variable']
        scaled_var_imp_df = pd.concat([scaled_var_imp_df, df_tmp], axis =1, sort = False)

    # sum rows by index, NaNs are consdered as zeros
    #Total sum per row: 
    scaled_var_imp_df.loc[:,'Total'] = scaled_var_imp_df.sum(axis=1)
    
    # Scaled from range 0 - 1
    scaled_var_imp_df.loc[:,'Total']=(scaled_var_imp_df['Total']-scaled_var_imp_df['Total'].min())/(scaled_var_imp_df['Total'].max()-scaled_var_imp_df['Total'].min())

    # Drop variables by a fs_threshold condition
    scaled_var_imp_df = scaled_var_imp_df[scaled_var_imp_df.Total > fs_threshold]

    # Sort by 'Total' values
    scaled_var_imp_df_sorted = scaled_var_imp_df.sort_values(by=['Total'], ascending = False)
    
    # Print out the table
    print(scaled_var_imp_df_sorted['Total'])

    # Plot and save bar chart
    plt.rcParams['xtick.labelsize'] = 12
    ax = scaled_var_imp_df_sorted.plot.bar(y='Total', rot=90, figsize=(16,12))
    plt.tight_layout()
    plt.savefig('FS_result_h2o.pdf', format='pdf', dpi=1200)

if ("StackedEnsemble" in model_key) is False:
    # get varimp_df
    varimp_df = aml.leader.varimp(use_pandas = True).iloc[:,[0,2]]
    scaled_var_imp_df = varimp_df

    # Drop variables by a fs_threshold condition
    scaled_var_imp_df = scaled_var_imp_df[scaled_var_imp_df.scaled_importance > fs_threshold]

    # Sort by 'scaled_importance' values
    scaled_var_imp_df_sorted = scaled_var_imp_df.sort_values(by=['scaled_importance'], ascending = False)
    
    # Print out the table
    print(scaled_var_imp_df_sorted)

    # Plot and save bar chart
    plt.rcParams['xtick.labelsize'] = 8
    ax = scaled_var_imp_df_sorted.plot.bar(y='scaled_importance', x = 'variable', rot=90, figsize=(16,12))
    plt.tight_layout()
    plt.savefig('FS_result_h2o.pdf', format='pdf', dpi=1200)

# Perform k-fold cv 
# split on train - test dataset by group - according to no_folds
gkf = GroupKFold(n_splits=no_folds)
cv_fold = 0

for train_index, test_index in gkf.split(X, y, groups=groups):
    cv_fold += 1
    print("CV fold: ", cv_fold)
    # print("Train Index: ", train_index)
    # print("Test Index: ", test_index, "\n")
    #print('Groups: ', groups,'\n')

    trainX_data = X.loc[train_index]
    trainy_data = y.loc[train_index]

    testX_data = X.loc[test_index]
    testy_data = y.loc[test_index]

    # Save original 10cv folds with all features
    train_set = pd.concat([trainX_data, trainy_data, groups], axis=1, join='inner')
    test_set = pd.concat([testX_data, testy_data, groups], axis=1, join='inner')

    # generate a file name based on the id and record and save orig 10cv datasets
    file_name_train = "10cv_orig_" + str(core_filename) + "_no" + str(cv_fold) + ".txt"
    file_name_test = "t-10cv_orig_" + str(core_filename) + "_no" + str(cv_fold) + ".txt"

    train_set.to_csv(r'./10cv_orig/' + file_name_train, index=False, sep="\t")
    test_set.to_csv(r'./10cv_orig/' + file_name_test, index=False, sep="\t")
    # print(model_key)
    if ('StackedEnsemble' in model_key) is True:
         # Remove features that score below threshold
         trainX_data = trainX_data[scaled_var_imp_df.index.tolist()]
         # trainy_data stays the same
         testX_data = testX_data[scaled_var_imp_df.index.tolist()]
    elif ('StackedEnsemble' in model_key) is False:
        # Remove features that score below threshold
        trainX_data = trainX_data[scaled_var_imp_df['variable']]
        # trainy_data stays the same
        testX_data = testX_data[scaled_var_imp_df['variable']]
        # testy_data stays the same

    #functionality to manually add features, eg. 'Time_min' in dissolution profiles
    if len(include_features) > 0:
        include_features_df_train = X.loc[train_index]
        include_features_df_test = X.loc[test_index]
        include_features_df_train = include_features_df_train[include_features]
        include_features_df_test = include_features_df_test[include_features]

        trainX_data = pd.concat([include_features_df_train, trainX_data], axis = 1)
        testX_data = pd.concat([include_features_df_test, testX_data], axis = 1)
        trainX_data = trainX_data.loc[:,~trainX_data.columns.duplicated()]
        testX_data = testX_data.loc[:,~testX_data.columns.duplicated()]

    train_set = pd.concat([trainX_data, trainy_data], axis=1)
    test_set = pd.concat([testX_data, testy_data], axis=1)

    ncols=train_set.shape[1]-1
    nrows=train_set.shape[0]

    print('nrows for' + aml2_name + ' project train dataset = ',nrows)
    print('ncols for train dataset = ',ncols)

    # save datasets after feature selection
    file_name_train = "10cv_" + str(core_filename)+ "_FS_to_" + str(ncols) + "_in" +"_no"+str(cv_fold)+".txt"
    file_name_test = "t-10cv_"+str(core_filename)+ "_FS_to_" + str(ncols) + "_in" +"_no"+str(cv_fold)+".txt"

    train_set.to_csv(r'./10cv_FS/' + file_name_train, index=False, sep="\t")
    test_set.to_csv(r'./10cv_FS/' + file_name_test, index = False, sep="\t")
# split loop end

# Load testing data in a loop and make folds based on them 
# 1) List all files with pattern 't-*.txt' in ./10cv_orig
all_filenames = [i for i in glob.glob('./10cv_FS/t-*.txt')]
# 2) Sort list of filenames from 1 to 10
all_filenames.sort(key = lambda x: int(x.split('_no')[1].split('.')[0]))
# 3) read all files in a list into a data_frame and make indicies for each t-file
df_new_approach = pd.concat([pd.read_csv(all_filenames[index], header = [0], sep = '\t', engine = 'python').assign(Fold_no=index+1) for index in range(len(all_filenames))])

# index of the output column
y_idx = df_new_approach.columns[df_new_approach.shape[1]-2]
training_frame = h2o.H2OFrame(df_new_approach)

# assign fold column name
assignment_type = 'Fold_no'

# set new AutoML options
aml_10cv = H2OAutoML(max_runtime_secs = h2o_max_runtime_secs_10cv,
                     seed = my_seed_FS,
                     project_name = aml2_name,
                     nfolds = no_folds,
                     export_checkpoints_dir = str(my_export_dir),
                     keep_cross_validation_predictions = my_keep_cross_validation_predictions,
                     keep_cross_validation_models = my_keep_cross_validation_models,
                     keep_cross_validation_fold_assignment = my_keep_cross_validation_fold_assignment,
                     verbosity = 'info',
                     sort_metric = 'RMSE')

# train AutoML with fold_column!
aml_10cv.train(y = y_idx, training_frame = training_frame, fold_column = assignment_type)

aml_10cv.leaderboard

aml_10cv.leader

# save h2o model
print('Saving leader h2o model in ./model_10cv and ./test_external')
my_10cv_model_path = h2o.save_model(aml_10cv.leader, path = './model_10cv', force = True)

print('')
print('The final model afer k-fold cv is located at: ')
print(str(my_10cv_model_path))
print('')

h2o.save_model(aml_10cv.leader, path = './test_external', force = True)

# Download POJO or MOJO
if save_pojo_or_mojo is True:
    if aml_10cv.leader.have_pojo is True:
        aml_10cv.leader.download_pojo(get_genmodel_jar = True, path = './pojo_or_mojo_10cv')
    if aml_10cv.leader.have_mojo is True:
        aml_10cv.leader.download_mojo(get_genmodel_jar = True, path = './pojo_or_mojo_10cv')

# get the models id
model_ids = list(aml_10cv.leaderboard['model_id'].as_data_frame().iloc[:,0])
# get the best model
m = h2o.get_model(aml_10cv.leader.key)
print('Leader model: ')
print(m.key)


if ("StackedEnsemble" in aml_10cv.leader.key) is True:
    # get the metalearner name
    se_meta_model = h2o.get_model(m.metalearner()['name'])

    my_se_meta_model_path = h2o.save_model(se_meta_model, path = './model_10cv')
    print('')
    print('The meta model of the best model is located at: ')
    print(str(my_se_meta_model_path))
    print('')

    h2o_cv_data = se_meta_model.cross_validation_holdout_predictions()
    pred_obs = h2o_cv_data.cbind([training_frame[training_frame.col_names[len(training_frame.col_names)-2]],training_frame['Fold_no'], training_frame['Time_Days']])

    # get a list of models - save and print out
    model_list = []

    print('Saving constituents of the StackedEnsemble')
    for model in m.params['base_models']['actual']:
        model_list.append(model['name'])
        my_tmp_model_path = h2o.save_model(h2o.get_model(str(model['name'])), path = './model_10cv')
        print(str(my_tmp_model_path))

    print('Stacked Ensemble model contains: ')
    print(model_list)


if ("StackedEnsemble" in aml_10cv.leader.key) is False:
    h2o_cv_data = m.cross_validation_holdout_predictions()
    pred_obs = h2o_cv_data.cbind([training_frame[training_frame.col_names[len(training_frame.col_names)-2]], training_frame['Fold_no'], training_frame['Time_Days']])

# Get the formulation indicies after split to 10-cv folds
orig_10cv_filenames = [i for i in glob.glob('./10cv_orig/t-*.txt')]
# 2) Sort list of filenames from 1 to 10
orig_10cv_filenames.sort(key = lambda x: int(x.split('_no')[1].split('.')[0]))
# 3) read all files in a list into a data_frame and make indicies for each t-file
df_orig_10cv = pd.concat([pd.read_csv(orig_10cv_filenames[index], header = [0], sep = '\t', engine = 'python').assign(Fold_no=index+1) for index in range(len(orig_10cv_filenames))])

tmp_data = df_orig_10cv.drop(['Q_perc'],axis = 1)
tmp_formulation = df_orig_10cv[['Formulation_no', 'Time_Days', 'Q_perc']]
tmp_data_h2o = h2o.H2OFrame(tmp_data)
tmp_predict = aml_10cv.predict(tmp_data_h2o)
tmp_predict = tmp_predict.as_data_frame(use_pandas = True)
tmp_formulation = tmp_formulation.assign(predict= pd.Series(tmp_predict['predict']).values)
df2 = tmp_formulation

df2.head(24)

df2_grouped = df2.groupby(['Formulation_no'])
fig2, ax2 = plt.subplots(23,3, figsize=(12,48), sharex='all', sharey = 'all', squeeze = False)

for i in df2_grouped.groups:
    tmp_df = df2_grouped.get_group(i)
    rmse_tmp = sqrt(mean_squared_error(tmp_df['Q_perc'], tmp_df['predict']))
    plt.subplot(23,3,i)
    plt.subplots_adjust(hspace = 0.4)
    plt.ylim(0,100)
    plt.plot(tmp_df['Time_Days'],tmp_df['Q_perc'], color='blue')
    plt.plot(tmp_df['Time_Days'],tmp_df['predict'], color='red')
    plt.text(0, 105, str('Form no ' + str(i) + ', RMSE = ' + str(round(rmse_tmp,2))), weight="bold")
    plt.legend()
# SAVE figure
plt.savefig('Q_perc_obs_pred_result_training_error_h2o.pdf', format='pdf', dpi=1200)

pred_obs_df = pred_obs.as_data_frame()
pred_obs_df = pred_obs_df.assign(Formulation_no= pd.Series(df_orig_10cv['Formulation_no']).values)
pred_obs_df = pred_obs_df.assign(Time_Days= pd.Series(df_orig_10cv['Time_Days']).values)

df3 = pred_obs_df
df3_grouped = df3.groupby(['Formulation_no'])

df3.head(24)

fig2, ax2 = plt.subplots(23,3, figsize=(12,48), sharex='all', sharey = 'all', squeeze = False)

for i in df3_grouped.groups:
    tmp_df = df3_grouped.get_group(i)
    rmse_tmp = sqrt(mean_squared_error(tmp_df['Q_perc'], tmp_df['predict']))
    plt.subplot(23,3,i)
    plt.subplots_adjust(hspace = 0.4)
    plt.ylim(0,100)
    plt.plot(tmp_df['Time_Days'], tmp_df['Q_perc'], color='blue')
    plt.plot(tmp_df['Time_Days'], tmp_df['predict'], color='red')
    plt.text(0, 105, str('Form no ' + str(i) + ', RMSE = ' + str(round(rmse_tmp,2))), weight="bold")
    plt.legend()
# SAVE figure
plt.savefig('Q_perc_obs_pred_result_10cv_error_h2o.pdf', format='pdf', dpi=1200)

trainX = training_frame.as_data_frame()
feature_names = list(trainX.columns)

# We will load saved model rather than extract the best model
# h2o_bst_model = aml_10cv.leader
h2o_bst_model = h2o.load_model('./model_10cv/StackedEnsemble_BestOfFamily_AutoML_20200108_112101')
h2o_wrapper = H2OPredWrapper(h2o_bst_model,feature_names)
# This is the core code for Shapley values calculation
# h2o.no_progress()
explainer = shap.KernelExplainer(h2o_wrapper.predict, shap.kmeans(trainX, 50))
# It took about 20 hours to calculate
# shap_values = explainer.shap_values(trainX, l1_reg = 'aic')
# pickle.dump(explainer, open("shap_explainer.p", "wb" ))
# pickle.dump(shap_values, open("shap_values.p", "wb" ))
# explainer = pickle.load(open("./shap_explainer.p", "rb"))
shap_values = pickle.load(open("./shap_values.p", "rb"))
# initialize js for SHAP
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[9,:], trainX.iloc[9,:])
shap.force_plot(explainer.expected_value, shap_values, trainX)
shap.dependence_plot('Mean_part_size', shap_values, trainX, interaction_index='PLGA_visc')
shap.dependence_plot('Prod_method', shap_values, trainX, interaction_index='Mean_part_size')
shap_summary = shap.summary_plot(shap_values, trainX)
shap_summary_bar = shap.summary_plot(shap_values, trainX, plot_type="bar")
