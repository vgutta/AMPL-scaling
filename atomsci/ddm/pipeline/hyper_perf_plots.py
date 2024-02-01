"""
Functions for visualizing hyperparameter performance. These functions work with 
a dataframe of model performance metrics and hyperparameter specifications from
compare_models.py. For models on the tracker, use get_multitask_perf_from_tracker().
For models in the file system, use get_filesystem_perf_results().
By Amanda P. 7/19/2022
"""
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Create an array with the colors you want to use
colors = ["#7682A4","#A7DDD8","#373C50","#694691","#BE2369","#EB1E23","#6EC8BE","#FFC30F",]
# Set your custom color palette
pal=sns.color_palette(colors)
sns.set_palette(pal)
plt.rcParams.update({"figure.dpi": 96})

from atomsci.ddm.pipeline import parameter_parser as pp

# get all possible things to plot from parameter parser
parser=pp.get_parser()
d=vars(pp.get_parser().parse_args())
keywords=['AttentiveFPModel','GCNModel','GraphConvModel','MPNNModel','PytorchMPNNModel','rf_','xgb_']
plot_dict={}
for word in keywords:
    tmplist=[x for x in d.keys() if x.startswith(word)]
    if word=='rf_': word='RF'
    elif word=='xgb_':word='xbgoost'
    plot_dict[word]=tmplist
plot_dict['general']=['model_type','features','splitter']#'ecfp_radius',
plot_dict['NN']=['plot_dropout','learning_rate','num_nodes','num_layers','best_epoch','max_epochs']

# list score types
regselmets=[
 'r2_score',
 'mae_score',
 'rms_score',
]
classselmets = [
 'roc_auc_score',
 'prc_auc_score',
 'precision',
 'recall_score',
 'npv',
 'accuracy_score',
 'kappa',
 'matthews_cc',
 'bal_accuracy',
]

def get_score_types():
    """
    Helper function to show score type choices.
    """
    print("Classification metrics: ", classselmets)
    print("Regression metrics: ", regselmets)

def _prep_perf_df(df):
    """
    This function splits columns that contain lists into individual columns to 
    use for plotting later.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a 
        hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or 
        get_filesystem_perf_results().
        
    Returns:
        perf_track_df (pd.DataFrame): a new df with modified and extra columns.
    """
    perf_track_df=df.copy()
    if 'NN' in perf_track_df.model_type.unique():
        perf_track_df['plot_dropout'] = perf_track_df.dropouts.astype(str).str.strip('[]').str.split(pat=',',n=1, expand=True)[0]
        perf_track_df['plot_dropout'] = perf_track_df.plot_dropout.astype(float)
        perf_track_df['layer_sizes'] = perf_track_df.layer_sizes.astype(str).str.strip('[]')
        cols=['dummy_nodes_1','dummy_nodes_2','dummy_nodes_3']
        tmp=perf_track_df.layer_sizes.str.split(pat=',', expand=True).astype(float)
        n=len(tmp.columns)
        perf_track_df[cols[0:n]]=tmp
        perf_track_df['num_layers'] = n-perf_track_df[cols[0:n]].isna().sum(axis=1)
        perf_track_df[cols[0:n]]=perf_track_df[cols[0:n]].fillna(value=1).astype(int)
        perf_track_df['num_nodes']=perf_track_df[cols[0:n]].product(axis=1)
        perf_track_df.num_nodes=perf_track_df.num_nodes.astype(float)
        perf_track_df=perf_track_df.drop(columns=cols[0:n])
        perf_track_df.loc[perf_track_df.model_type != "NN", 'layer_sizes']=np.nan
        perf_track_df.loc[perf_track_df.model_type != "NN", 'num_layers']=np.nan
        perf_track_df.loc[perf_track_df.model_type != "NN", 'num_nodes']=np.nan
        perf_track_df.loc[perf_track_df.model_type != "NN", 'plot_dropout']=np.nan
    
    return perf_track_df

def plot_train_valid_test_scores(df, prediction_type='regression'):
    """
    This function plots line plots of performance scores based on their partitions.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a 
        hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or 
        get_filesystem_perf_results().
        
        scoretype (str): the score type you want to use. Valid options can be found in
        hpp.classselmets or hpp.regselmets.  
    """
    sns.set_context('notebook')
    perf_track_df=df.copy().reset_index(drop=True)

    if prediction_type=='regression':
        selmets=regselmets
    elif prediction_type=='classification':
        selmets=classselmets

    with sns.axes_style("ticks"):
        fig, ax = plt.subplots(1,len(selmets), figsize=(5*len(selmets),5))
        for i, scoretype in enumerate(selmets):
            plot_df=perf_track_df[[f"best_train_{scoretype}",f"best_valid_{scoretype}",f"best_test_{scoretype}"]]
            plot_df=plot_df.sort_values(f"best_valid_{scoretype}")
            ax[i].plot(plot_df.T);
            ax[i].set_ylim(plot_df.min().min()-.1,1)
            ax[i].tick_params(rotation=15)
            ax[i].set_title(scoretype)
        fig.suptitle(f"Model performance by partition");
        plt.tight_layout()

    
### the following 3 plots are originally from Amanda M.
def plot_rf_perf(df, scoretype='r2_score',subset='valid'):
    """
    This function plots scatterplots of performance scores based on their RF hyperparameters.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a 
        hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or 
        get_filesystem_perf_results().
        
        scoretype (str): the score type you want to use. Valid options can be found in
        hpp.classselmets or hpp.regselmets.
        
        subset (str): the subset of scores you'd like to plot from 'train', 'valid' and 'test'.
    """
    sns.set_context('poster')
    perf_track_df=df.copy().reset_index(drop=True)
    plot_df=perf_track_df[perf_track_df.model_type=='RF'].copy()
    winnertype= f'best_{subset}_{scoretype}'
    
    if len(plot_df)>0:
        feat1 = 'rf_estimators'; feat2 = 'rf_max_features'; feat3 = 'rf_max_depth'
        plot_df[f'{feat1}_cut']=pd.cut(plot_df[feat1], 5, precision=0)
        plot_df[f'{feat2}_cut']=pd.cut(plot_df[feat2], 5, precision=0)
        hue=feat3
        plot_df = plot_df.sort_values([f'{feat1}_cut', f'{feat2}_cut',feat3], ascending=True)
        plot_df.loc[:,f'{feat1}/{feat2}'] = ['%s / %s' % (mf,est) for mf,est in zip(plot_df[f'{feat1}_cut'], plot_df[f'{feat2}_cut'])]
        with sns.axes_style("whitegrid"):
            fig,ax = plt.subplots(1,figsize=(40,15))
            if plot_df[hue].nunique()<13:
                palette=sns.cubehelix_palette(plot_df[hue].nunique())
            else:
                palette=sns.cubehelix_palette(as_cmap=True)
            sns.scatterplot(x=f'{feat1}/{feat2}', y=winnertype, hue=hue, palette=palette, data=plot_df, ax=ax)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.xticks(rotation=30, ha='right')
            ax.set_title(f'RF model performance');
    else: print("There are no RF models in this set.")

        
def plot_nn_perf(df, scoretype='r2_score',subset='valid'):
    """
    This function plots scatterplots of performance scores based on their NN hyperparameters.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a 
        hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or 
        get_filesystem_perf_results().
        
        scoretype (str): the score type you want to use. Valid options can be found in
        hpp.get_score_types().
        
        subset (str): the subset of scores you'd like to plot from 'train', 'valid' and 'test'.
    """
    sns.set_context('poster')
    perf_track_df=_prep_perf_df(df).reset_index(drop=True)
    plot_df=perf_track_df[perf_track_df.model_type=='NN'].copy()
    winnertype= f'best_{subset}_{scoretype}'
    
    if len(plot_df)>0:
        feat1 = 'num_nodes'; feat2 = 'learning_rate'; feat3 = 'plot_dropout'
        plot_df[f'{feat1}_cut']=pd.cut(plot_df[feat1],5)
        plot_df[f'{feat2}_cut']=pd.cut(plot_df[feat2],5)
        plot_df = plot_df.sort_values([f'{feat1}_cut', f'{feat2}_cut',feat3], ascending=True)
        for feat in [feat1,feat2]:
            bins=['{:.2e}'.format(x) for x in pd.cut(plot_df[feat],5,retbins=True)[1]]
            binstrings=[]
            for i,bin in enumerate(bins):
                try:binstrings.append(f'({bin}, {bins[i+1]}]')
                except:pass
            nncmap=dict(zip(plot_df[f'{feat}_cut'].dtype.categories.tolist(),binstrings))
            plot_df[f'{feat}_cut']=plot_df[f'{feat}_cut'].map(nncmap)
        hue=feat3
        plot_df[f'{feat1}/{feat2}'] = ['%s / %s' % (mf,est) for mf,est in zip(plot_df[f'{feat1}_cut'], plot_df[f'{feat2}_cut'])]
        with sns.axes_style("whitegrid"):
            fig,ax = plt.subplots(1,figsize=(40,15))
            if plot_df[hue].nunique()<13:
                palette=sns.cubehelix_palette(plot_df[hue].nunique())
            else:
                palette=sns.cubehelix_palette(as_cmap=True)
            sns.scatterplot(x=f'{feat1}/{feat2}', y=winnertype, hue=hue,
                            palette=palette, 
                            data=plot_df, ax=ax)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.xticks(rotation=30, ha='right')
            ax.set_title(f'NN model performance');
    else: print("There are no NN models in this set.")
    return plot_df
        
def plot_xg_perf(df, scoretype='r2_score',subset='valid'):
    """
    This function plots scatterplots of performance scores based on their XG hyperparameters.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a 
        hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or 
        get_filesystem_perf_results().
        
        scoretype (str): the score type you want to use. Valid options can be found in
        hpp.classselmets or hpp.regselmets.
        
        subset (str): the subset of scores you'd like to plot from 'train', 'valid' and 'test'.
    """
    sns.set_context('poster')
    perf_track_df=df.copy().reset_index(drop=True)
    plot_df=perf_track_df[perf_track_df.model_type=='xgboost']
    winnertype= f'best_{subset}_{scoretype}'
    if len(plot_df)>0:
        feat1 = 'xgb_learning_rate'; feat2 = 'xgb_gamma'
        hue=feat2
        plot_df = plot_df.sort_values([feat1, feat2])
        #plot_df[f'{feat1}/{feat2}'] = ['%s / %s' % (mf,est) for mf,est in zip(plot_df[feat1], plot_df[feat2])]
        with sns.axes_style("whitegrid"):
            fig,ax = plt.subplots(1,figsize=(40,15))
            if plot_df[hue].nunique()<13:
                # palette=sns.cubehelix_palette(plot_df[hue].nunique())
                sns.scatterplot(x=feat1, y=winnertype, hue=hue, data=plot_df, ax=ax1)
            else:
                palette=sns.cubehelix_palette()
                sns.scatterplot(x=feat1, y=winnertype, hue=hue, 
                                palette=palette, 
                                data=plot_df, ax=ax1)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.xticks(rotation=30, ha='right')
            ax.set_title(f'XGboost model performance');
    else: print('There are no XGBoost models in this set.')

def plot_hyper_perf(df, scoretype='r2_score', subset='valid', model_type='general'):
    """
    This function plots boxplots or scatter plots of performance scores based on their hyperparameters.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a 
        hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or 
        get_filesystem_perf_results().
        
        scoretype (str): the score type you want to use. Valid options can be found in
        hpp.classselmets or hpp.regselmets.
        
        subset (str): the subset of scores you'd like to plot from 'train', 'valid' and 'test'.

        model_type (str): the type of model you want to visualize. Options include 'general' (features and splits), 'RF', 'NN', 'xgboost', and limited functionality for 'AttentiveFPModel', 'GCNModel', 'GraphConvModel', 'MPNNModel', and 'PytorchMPNNModel'.
    """

    sns.set_context('notebook')
    perf_track_df=_prep_perf_df(df).reset_index(drop=True)
    winnertype= f'best_{subset}_{scoretype}'

    feats=plot_dict[model_type]
    ncols=3
    nrows=int(np.ceil(len(feats)/ncols))

    helix_dict={'NN':(0,0.40),'RF':(0,2.00),'xgboost':(0,2.75),'general':(60,0.2)}
    
    fig, ax = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*4))
    ax=ax.ravel()
    for i, feat in enumerate(feats):
        try: rot,start=helix_dict[model_type]
        except: rot,start=(-0.2,0)
        if feat in perf_track_df.columns:    
            if perf_track_df[feat].nunique()>12:
                sns.scatterplot(x=feat, y=winnertype, data=perf_track_df, ax=ax[i])
            else:       
                sns.boxplot(x=feat,y=winnertype,palette=sns.cubehelix_palette(perf_track_df[feat].nunique(), rot=rot,start=start,), data=perf_track_df, ax=ax[i])
        ax[i].tick_params(axis='x', rotation=30)
        ax[i].set_xlabel(feat)
    fig.suptitle(f'{model_type} hyperparameter performance')
    plt.tight_layout()
    return perf_track_df

def plot_rf_nn_xg_perf(df, scoretype='r2_score',subset='valid'):
    """
    This function plots boxplots of performance scores based on their hyperparameters including
    RF, NN and XGBoost parameters as well as feature types, model types and ECFP radius.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a 
        hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or 
        get_filesystem_perf_results().
        
        scoretype (str): the score type you want to use. Valid options can be found in
        hpp.classselmets or hpp.regselmets.
        
        subset (str): the subset of scores you'd like to plot from 'train', 'valid' and 'test'.
    """
    sns.set_context('notebook')
    perf_track_df=_prep_perf_df(df).reset_index(drop=True)
    winnertype= f'best_{subset}_{scoretype}'
    
    nfeats=3
    feat1='plot_dropout'; feat2='learning_rate'; feat3='num_nodes'
    feat4='num_layers'; feat5='rf_max_depth'; feat6='rf_max_features'
    feat7='rf_estimators'; feat8='xgb_gamma'; feat9='xgb_learning_rate'
    feat10='features'; feat11='model_type'; feat12=f'best_test_{scoretype}'; feat13='ecfp_radius'
    
    plotdf2=perf_track_df
    fig, ax = plt.subplots(4,3, figsize=(16,12))
    if 'NN' in perf_track_df.model_type.unique():
        sns.boxplot(x=feat1, y=winnertype, palette=sns.cubehelix_palette(len(plotdf2[feat1].unique()), rot=0, start=0.40), data=plotdf2,    ax=ax[0,0]); ax[0,0].tick_params(rotation=0);  ax[0,0].set_xlabel('NN dropouts')
        sns.boxplot(x=feat2, y=winnertype, palette=sns.cubehelix_palette(len(plotdf2[feat2].unique()), rot=0, start=0.40), data=plotdf2,    ax=ax[0,1]); ax[0,1].tick_params(rotation=30); ax[0,1].set_xlabel('NN learning rate')#ax[0,1].legend_.remove(); ax[0,1].title.set_text(f"Hyperparameters colored by {feat1}")
        plotdf=perf_track_df[perf_track_df[feat3]>0]
        sns.boxplot(x=feat3, y=winnertype, palette=sns.cubehelix_palette(len(plotdf[feat3].unique()), rot=0, start=0.40), data=plotdf,    ax=ax[0,2]); ax[0,2].tick_params(rotation=30); ax[0,2].set_xlabel('NN number of parameters in hidden layers')#ax[0,2].legend_.remove()#(bbox_to_anchor=(1,1), title=feat1)#, prop={'size': 12})
        sns.boxplot(x=feat4, y=winnertype, palette=sns.cubehelix_palette(len(plotdf2[feat4].unique()), rot=0, start=0.40), data=plotdf2,    ax=ax[1,0]); ax[1,0].tick_params(rotation=0);  ax[1,0].set_xlabel('NN number of layers')#ax[1,0].legend_.remove(); ax[1,0].tick_params(rotation=45)
    if 'xgboost' in perf_track_df.model_type.unique():
        sns.boxplot(x=feat8, y=winnertype, palette=sns.cubehelix_palette(len(plotdf2[feat8].unique()), rot=0, start=2.75), data=plotdf2,    ax=ax[1,1]); ax[1,1].tick_params(rotation=0);  ax[1,1].set_xlabel('XGBoost gamma')#ax[1,1].title.set_text(f"Hyperparameters colored by {feat2}")
        sns.boxplot(x=feat9, y=winnertype, palette=sns.cubehelix_palette(len(plotdf2[feat9].unique()), rot=0, start=2.75), data=plotdf2,    ax=ax[1,2]); ax[1,2].tick_params(rotation=0);  ax[1,2].set_xlabel('XGBoost learning rate')#ax[1,2].legend_.remove()#(bbox_to_anchor=(1,1), title=feat2)
    if 'RF' in perf_track_df.model_type.unique():
        sns.boxplot(x=plotdf2.loc[~plotdf2[feat7].isna(),feat7].astype(int), y=winnertype, palette=sns.cubehelix_palette(len(plotdf2[feat7].unique()), rot=0, start=2.00), data=plotdf2,    ax=ax[2,0]); ax[2,0].tick_params(rotation=0); ax[2,0].set_xlabel('RF number of trees')#ax[2,0].legend_.remove(); ax[2,0].tick_params(rotation=45)
        try:
            sns.boxplot(x=plotdf2.loc[~plotdf2[feat5].isna(),feat5].astype(int), y=winnertype, palette=sns.cubehelix_palette(len(plotdf2[feat5].unique()), rot=0, start=2.00), data=plotdf2,    ax=ax[2,1]); ax[2,1].tick_params(rotation=0)
        except: pass
        ax[2,1].set_xlabel('RF max depth')#ax[2,1].legend_.remove(); ax[2,1].title.set_text(f"Hyperparameters colored by {feat3}")
        sns.boxplot(x=plotdf2.loc[~plotdf2[feat6].isna(),feat6].astype(int), y=winnertype, palette=sns.cubehelix_palette(len(plotdf2[feat6].unique()), rot=0, start=2.00), data=plotdf2,    ax=ax[2,2]); ax[2,2].tick_params(rotation=0); ax[2,2].set_xlabel('RF max features per node')#ax[2,2].legend(bbox_to_anchor=(1,1), title=feat3);
    #general
    plotdf2=plotdf2.sort_values(feat10)
    sns.boxplot(x=feat11, y=winnertype, palette=sns.cubehelix_palette(len(plotdf2[feat11].unique()), rot=60, start=0.20), data=plotdf2,  ax=ax[3,1]); ax[3,1].tick_params(rotation=0);  ax[3,1].set_xlabel('Model type')#ax[2,1].legend_.remove(); ax[2,1].title.set_text(f"Hyperparameters colored by {feat3}")
    sns.boxplot(x=feat10, y=winnertype, palette=sns.cubehelix_palette(len(plotdf2[feat10].unique()), rot=60, start=0.20), data=plotdf2,  ax=ax[3,0]); ax[3,0].tick_params(rotation=0);  ax[3,0].set_xlabel('Featurization type');ax[3,0].set_xticklabels( ax[3,0].get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor' )#ax[2,0].legend_.remove(); 
    if 'ecfp_radius' in perf_track_df.columns:
        sns.boxplot(x=feat13, y=winnertype, palette=sns.cubehelix_palette(len(plotdf2[feat13].unique()), rot=60, start=0.20), data=plotdf2,  ax=ax[3,2])
    ax[3,2].tick_params(rotation=0);  ax[3,2].set_xlabel('ECFP radius')#ax[2,1].legend_.remove(); ax[2,1].title.set_text(f"Hyperparameters colored by {feat3}")
    # sns.scatterplot(x=feat12, y=winnertype,palette=sns.cubehelix_palette(len(plotdf2[feat12].unique()),rot=0, start=0.20),data=plotdf2, ax=ax[3,2]); ax[3,2].tick_params(rotation=0);  ax[3,2].set_xlabel(f'{feat12}')#ax[2,2].legend(bbox_to_anchor=(1,1), title=feat3);

    plt.tight_layout()
    fig.suptitle(f"Effect of hyperparameter tuning on model performance", y=1.01);

    
def plot_split_perf(df, scoretype='r2_score',subset='valid'):
    """
    This function plots boxplots of performance scores based on the splitter type.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a 
        hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or 
        get_filesystem_perf_results().
        
        scoretype (str): the score type you want to use. Valid options can be found in
        hpp.classselmets or hpp.regselmets.
        
        subset (str): the subset of scores you'd like to plot from 'train', 'valid' and 'test'.
    """
    sns.set_context("notebook")    
    perf_track_df=_prep_perf_df(df).reset_index(drop=True)
    winnertype= f'best_{subset}_{scoretype}'

    if scoretype in regselmets:
        selmets=regselmets
    elif scoretype in classselmets:
        selmets=classselmets
        
    plot_df=perf_track_df
    plot_df=plot_df.sort_values('features')
    with sns.axes_style("ticks"):
        fig, axes = plt.subplots(1,len(selmets), figsize=(5*len(selmets),5))
        for i, ax in enumerate(axes.flat):
            if i==len(axes.flat)-1:
                legend=True
            else:
                legend=False
            selection_metric = f'best_{subset}_{selmets[i]}'
            sns.boxplot(x="features", y=selection_metric, # x="txptr_features" x="model_type"
                        hue='splitter', palette = sns.color_palette(colors[0:plot_df.features.nunique()]), #showfliers=False, 
                          legend=legend,
                        data=plot_df, ax=ax);
            ax.set_xlabel('')
            ax.set_ylabel(selection_metric.replace(f'best_{subset}_',''))
            ax.set_xticks(ax.get_xticks()) # avoid warning by including this line
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor' )
            ax.set_title(selection_metric.replace(f'best_{subset}_',''))
            if legend: sns.move_legend(ax, loc=(1.01,0.5))
        plt.tight_layout()
        fig.suptitle('Effect of splitter on model performance', y=1.01)
        