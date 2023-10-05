from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import typing
from typing import Sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


class EmpLengthTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame, y:pd.DataFrame = None) -> pd.DataFrame:
        emp_length_dict = {
            '< 1 year': 0,
            '1 year': 1,
            '2 years': 2,
            '3 years': 3,
            '4 years': 4,
            '5 years': 5,
            '6 years': 6,
            '7 years': 7,
            '8 years': 8,
            '9 years': 9,
            '10+ years': 10
        }

        X = X.apply(lambda x: emp_length_dict[x])
        return X


def binary_to_labeled(value: int) -> str:
    '''Changes binary features to "Yes" (1) and "No" (0)'''
    if value == 0:
        return 'No'
    
    return 'Yes'


def multiple_countplots(df: pd.DataFrame, columns: Sequence[str], target_feature: str):
    '''
    Given a pandas DataFrame, list of columns labels and target feature 
    plots countplots for all given columns with hue set as target feature.
    '''
    fig, ax = plt.subplots(1 * ((len(columns)-1) // 3 + 1), 3, figsize=(12, 4 * ((len(columns)-1) // 3 + 1)))

    axs = ax.flatten()

    for i, col in enumerate(columns):
        counts = df.groupby([col, target_feature]).size()

        levels = [counts.index.levels[0], counts.index.levels[1]]
        new_index = pd.MultiIndex.from_product(levels, names=counts.index.names)

        counts = counts.reindex(new_index, fill_value=0)

        values = df[col].unique()
        bottom = [counts[value, 0] for value in values]
        top = [counts[value, 1] for value in values]

        axs[i].bar(values, bottom)
        axs[i].bar(values, top, bottom=bottom)
        
        for container in axs[i].containers:
            labels = [value.get_height() if value.get_height() > 0 else '' for value in container]
        
            axs[i].bar_label(container, labels=labels, label_type='center')

        for label in axs[i].get_xticklabels():
            label.set_rotation(30)

        axs[i].set_xlabel(col)

    plt.tight_layout()


def get_model_name(variable: object) -> str:
    '''Returns model name given a model'''
    return str(variable).split('(')[0]


def brute_force_models(X_train: pd.DataFrame, y_train: pd.Series, text_column: str, categorical_columns: Sequence[str], numerical_columns: Sequence[str], scalers: Sequence[object], encoders: Sequence[object], models: Sequence[object]) -> pd.DataFrame:
    '''
    This function tries every given combination of encoders, scalers and models.

    It needs to be provided with:
    train_X - training dataset
    train_y - target dataset
    categorical_columns - list of labels of categorical columns in train_X
    numerical_columns - list of labels of numerical columns in train_X
    scalers - list of scalers
    encoders - list of encoder
    models - list of models

    Returns a pandas DataFrame with all combinations and their accuracy, recall, precision and f1 score
    '''
    i = 0

    df = pd.DataFrame(columns=['encoder', 'scaler', 'model', 'accuracy', 'recall', 'precision', 'f1_score'])
    for encoder in encoders:
        for scaler in scalers:
            transformer = ColumnTransformer([
                ('hash', HashingVectorizer(), text_column),
                ('tfidf', TfidfVectorizer(), text_column),
                ('categorical', encoder, categorical_columns),
                ('numerical', scaler, numerical_columns)
            ])
            for model in models:
                pipe = make_pipeline(transformer, TruncatedSVD(), model)
                
                pipe.fit(X_train, y_train)

                scores = cross_validate(pipe, X_train, y_train, scoring=['accuracy', 'recall', 'precision', 'f1'])

                accuracy = np.mean(scores['test_accuracy'])
                recall = np.mean(scores['test_recall'])
                precision = np.mean(scores['test_precision'])
                f1 = np.mean(scores['test_f1'])

                list = [get_model_name(encoder), get_model_name(scaler), get_model_name(model), accuracy, recall, precision, f1]

                df.loc[len(df)] = list

                print(i, list)
                i += 1

    return df


def adjust_class(pred: Sequence[float], t: float) -> list[int]:
    '''Given a list of probabilities and a threshold, returns a list with binary classes'''
    return [1 if y >= t else 0 for y in pred]


def model_report(y_test: Sequence[float], y_pred: Sequence[float]):
    '''Given y_test and y_pred provides accuracy score, confusiom matrix and classification report'''
    print('Accuracy:\n\t', accuracy_score(y_test, y_pred), '\n')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred), '\n')
    print('Classification report:\n', classification_report(y_test, y_pred), '\n')


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    '''
    Plots precision recall curve
    '''
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", linewidth=2, label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", linewidth=2, label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


def plot_roc_curve(fpr, tpr, label):
    '''
    Plots ROC curve given False Positive Rate and True Positive Rate
    '''
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)

    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.show()


def remove_null_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    column_null_counts = df.isna().sum()
    columns = column_null_counts[column_null_counts < (1 - threshold) * len(df)].index
    return df[columns]


def read_columns_from_txt(path: str) -> list[str]:
    my_file = open(path, 'r')

    data = my_file.read()
    
    data_into_list = data.split('\n')
    my_file.close()
    return data_into_list


def shorten_titles(df: pd.DataFrame(), titles: list[str]) -> None:
    for title in titles:
        df.loc[df['emp_title'].str.contains(title), 'emp_title'] = title


def count_plot(df: pd.DataFrame, feature: str, hue: str = None) -> None:
    temp_df = df.copy()
    temp_df['constant'] = 1

    temp_df = pd.pivot_table(temp_df, index=feature, columns=hue, values='constant', aggfunc=np.size)

    fig, ax = plt.subplots()
    bottom = np.zeros(len(temp_df.index))

    for column in temp_df.columns:
        p = ax.bar(temp_df.index, temp_df[column], label=column, bottom=bottom)
        bottom += temp_df[column]

        ax.bar_label(p, label_type='center', fmt = '%d')

    ax.ticklabel_format(useOffset=False, style='plain', axis='y')
    ax.set_yticks(ax.get_yticks()[:-1], [f"{int(x):,}" for x in ax.get_yticks()[:-1]]);
    
    if hue:
        ax.legend()


def stacked_plot(df: pd.DataFrame, feature: str, hue: str) -> None:
    temp_df = pd.crosstab(index=df[feature],
                          columns=df[hue],
                          normalize='index').apply(lambda x: np.round(x * 100, 1))

    fig, ax = plt.subplots()
    bottom = np.zeros(len(temp_df.index))

    for column in temp_df.columns:
        p = ax.bar(temp_df.index, temp_df[column], label=column, bottom=bottom)
        bottom += temp_df[column]

        ax.bar_label(p, label_type = 'center', fmt = '%d')

    ax.ticklabel_format(useOffset=False, style='plain', axis='y')
    ax.set_yticks(ax.get_yticks()[:-1], [f"{int(x):,}" for x in ax.get_yticks()[:-1]]);

    if hue:
        ax.legend()


def drop_numerical_outliers(df: pd.DataFrame, z_thresh: int = 3) -> pd.DataFrame:
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh) \
        .all(axis=1)
        
    return df.drop(df.index[~constrains])


def plot_confusion_matrix(y_pred, y_test):
    matrix = confusion_matrix(y_pred, y_test)

    sns.heatmap(matrix / np.sum(matrix), annot=True, fmt='.2%', cmap='Blues')

    plt.title('Confusion Matrix')
    plt.xlabel('Real')
    plt.ylabel('Predicted')

    plt.show()