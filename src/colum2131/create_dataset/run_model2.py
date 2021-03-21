from base_model2 import Feature, generate_features

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
import scipy as sp
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, _document_frequency
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from category_encoders.target_encoder import TargetEncoder
import texthero as hero
import xfeat


def get_subtitle(df: pd.DataFrame):
    for axis in ['h', 'w', 't', 'd']:
        column_name = f'size_{axis}'
        size_info = df['sub_title'].str.extract(r'{} (\d*|\d*\.\d*)(cm|mm)'.format(axis)) # 正規表現を使ってサイズを抽出
        size_info = size_info.rename(columns={0: column_name, 1: 'unit'})
        size_info[column_name] = size_info[column_name].replace('', np.nan).astype(float) # dtypeがobjectになってるのでfloatに直す
        size_info[column_name] = size_info.apply(lambda row: row[column_name] * 10 if row['unit'] == 'cm' else row[column_name], axis=1) # 　単位をmmに統一する
        df[column_name] = size_info[column_name]
    return df

def creat_dataset(df: pd.DataFrame):
    df = df.merge(
        maker_df, left_on='principal_maker', right_on='name', how='left'
    ).drop(columns=['name', 'nationality'], axis=1)

    df = get_subtitle(df)

    return df

class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b  : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        """
        if not sp.sparse.issparse(X):
            X = sp.sparse.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.sparse.spdiags(idf, diags=0, m=n_features, n=n_features)

        doc_len = X.sum(axis=1)
        self._average_document_len = np.average(doc_len)

        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
            # preserve float family dtype
            X = sp.sparse.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.sparse.csr_matrix(X, dtype=np.float, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        doc_len = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]

        # In each row, repeat `doc_len` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(doc_len), sz)

        # Compute BM25 score only for non-zero elements
        nom = self.k1 + 1
        denom = X.data + self.k1 * (1 - self.b + self.b * rep / self._average_document_len)
        data = X.data * nom / denom

        X = sp.sparse.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X = X * self._idf_diag

        return X

def get_text_features_count(input_df: pd.DataFrame, text_col: str, n_comp=10, clean=True):
    vectorizer = make_pipeline(
        CountVectorizer(),
        make_union(
            TruncatedSVD(n_components=n_comp, random_state=42),
            NMF(n_components=n_comp, random_state=42),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=n_comp, random_state=42)),
            n_jobs=1)
        )
    
    X = input_df[text_col].fillna('')
    if clean == True:
        pipeline = [
            hero.preprocessing.fillna,
            hero.preprocessing.remove_digits,
            hero.preprocessing.remove_punctuation,
            hero.preprocessing.remove_diacritics,
            hero.preprocessing.remove_whitespace]
        hero.clean(X, pipeline)

    X = vectorizer.fit_transform(X).astype(np.float32)
    output_df = pd.DataFrame(X, columns=[
        f'{text_col}_count_svd_{i}' for i in range(n_comp)] + [
        f'{text_col}_count_nmf_{i}' for i in range(n_comp)] + [
        f'{text_col}_count_bm25_{i}' for i in range(n_comp)])

    return output_df


def get_text_features_tfidf(input_df: pd.DataFrame, text_col: str, n_comp=10, clean=True):
    vectorizer = make_pipeline(
        TfidfVectorizer(),
        make_union(
            TruncatedSVD(n_components=n_comp, random_state=42),
            NMF(n_components=n_comp, random_state=42),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=n_comp, random_state=42)),
            n_jobs=1)
        )
    
    X = input_df[text_col].fillna('')
    if clean == True:
        pipeline = [
            hero.preprocessing.fillna,
            hero.preprocessing.remove_digits,
            hero.preprocessing.remove_punctuation,
            hero.preprocessing.remove_diacritics,
            hero.preprocessing.remove_whitespace]
        hero.clean(X, pipeline)

    X = vectorizer.fit_transform(X).astype(np.float32)
    output_df = pd.DataFrame(X, columns=[
        f'{text_col}_tfidf_svd_{i}' for i in range(n_comp)] + [
        f'{text_col}_tfidf_nmf_{i}' for i in range(n_comp)] + [
        f'{text_col}_tfidf_bm25_{i}' for i in range(n_comp)])

    return output_df
"""
def get_sub_title(df: pd.DataFrame):
    tmp = []
    _dict = {'h':0, 'w': 1, 't':2, 'd':3}
    for data in df['sub_title'].fillna('').map(lambda x: x.split()):
        check = -1
        res = [float('nan')] * 4
        for i, c in enumerate(data):
            if c in _dict.keys():
                check = _dict[c]
            if 'm' in c:
                if len(c) <= 2: continue
                if 'mm' in c:
                    if c[:len(c)-2] == '78/54':
                        c = f'{78 / 54}mm'
                    res[check] = float(c[:len(c)-2])
                else:
                    res[check] = float(c[:len(c)-2])*10
        tmp.append(res)
    return pd.DataFrame(tmp, columns=['h', 'w', 't', 'd'])
"""

class ArtseriesidLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['art_series_id'].fillna(''))
        self.train['label_art_series_id'] = features[:len(train)]
        self.test['label_art_series_id'] = features[len(train):]

class ArtseriesidCountBasefeature(Feature):
    def create_features(self):
        self.train['count_art_series_id'] = train['art_series_id'].map(merge_df['art_series_id'].value_counts())
        self.test['count_art_series_id'] = test['art_series_id'].map(merge_df['art_series_id'].value_counts())

class TitleCount(Feature):
    def create_features(self):
        features = get_text_features_count(merge_df, 'title', 10, True)
        self.train = features[:len(train)]
        self.test = features[len(train):].reset_index(drop=True)

class TitleTfidf(Feature):
    def create_features(self):
        features = get_text_features_tfidf(merge_df, 'title', 10, True)
        self.train = features[:len(train)]
        self.test = features[len(train):].reset_index(drop=True)

class TitleBasefeature(Feature):
    def create_features(self):
        self.train['len_title'] = train['title'].fillna('').map(lambda x: len(x))
        self.test['len_title'] = test['title'].fillna('').map(lambda x: len(x))
        self.train['num_words_title'] = train['title'].fillna('').map(lambda x: len(x.split(' ')))
        self.test['num_words_title'] = test['title'].fillna('').map(lambda x: len(x.split(' ')))

class TitleLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['title'].fillna(''))
        self.train['label_title'] = features[:len(train)]
        self.test['label_title'] = features[len(train):]

class TitleCountBasefeature(Feature):
    def create_features(self):   
        self.train['count_title'] = train['title'].map(merge_df['title'].value_counts())
        self.test['count_title'] = test['title'].map(merge_df['title'].value_counts())

class DescriptionCount(Feature):
    def create_features(self):
        features = get_text_features_count(merge_df, 'description', 10, True)
        self.train = features[:len(train)]
        self.test = features[len(train):].reset_index(drop=True)

class DescriptionTfidf(Feature):
    def create_features(self):
        features = get_text_features_tfidf(merge_df, 'description', 10, True)
        self.train = features[:len(train)]
        self.test = features[len(train):].reset_index(drop=True)

class DescriptionBasefeature(Feature):
    def create_features(self):
        self.train['len_description'] = train['description'].fillna('').map(lambda x: len(x))
        self.test['len_description'] = test['description'].fillna('').map(lambda x: len(x))
        self.train['num_words_description'] = train['description'].fillna('').map(lambda x: len(x.split(' ')))
        self.test['num_words_description'] = test['description'].fillna('').map(lambda x: len(x.split(' ')))

class DescriptionLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['description'].fillna(''))
        self.train['label_description'] = features[:len(train)]
        self.test['label_description'] = features[len(train):]

class DescriptionCountBasefeature(Feature):
    def create_features(self):   
        self.train['count_description'] = train['description'].map(merge_df['description'].value_counts())
        self.test['count_description'] = test['description'].map(merge_df['description'].value_counts())

class LongtitleCount(Feature):
    def create_features(self):
        features = get_text_features_count(merge_df, 'long_title', 10, True)
        self.train = features[:len(train)]
        self.test = features[len(train):].reset_index(drop=True)

class LongtitleTfidf(Feature):
    def create_features(self):
        features = get_text_features_tfidf(merge_df, 'long_title', 10, True)
        self.train = features[:len(train)]
        self.test = features[len(train):].reset_index(drop=True)

class LongtitleBasefeature(Feature):
    def create_features(self):
        self.train['len_long_title'] = train['long_title'].fillna('').map(lambda x: len(x))
        self.test['len_long_title'] = test['long_title'].fillna('').map(lambda x: len(x))
        self.train['num_words_long_title'] = train['long_title'].fillna('').map(lambda x: len(x.split(' ')))
        self.test['num_words_long_title'] = test['long_title'].fillna('').map(lambda x: len(x.split(' ')))

class LongtitleLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['long_title'].fillna(''))
        self.train['label_long_title'] = features[:len(train)]
        self.test['label_long_title'] = features[len(train):]

class LongtitleCountBasefeature(Feature):
    def create_features(self):   
        self.train['count_long_title'] = train['long_title'].map(merge_df['long_title'].value_counts())
        self.test['count_long_title'] = test['long_title'].map(merge_df['long_title'].value_counts())

class PrincipalmakerLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['principal_maker'].fillna(''))
        self.train['label_principal_maker'] = features[:len(train)]
        self.test['label_principal_maker'] = features[len(train):]

class PrincipalmakerCountBasefeature(Feature):
    def create_features(self):   
        self.train['count_principal_maker'] = train['principal_maker'].map(merge_df['principal_maker'].value_counts())
        self.test['count_principal_maker'] = test['principal_maker'].map(merge_df['principal_maker'].value_counts())

class PrincipalorfirstmakerLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['principal_or_first_maker'].fillna(''))
        self.train['label_principal_or_first_maker'] = features[:len(train)]
        self.test['label_principal_or_first_maker'] = features[len(train):]

class PrincipalorfirstmakerCountBasefeature(Feature):
    def create_features(self):
        self.train['count_principal_or_first_maker'] = train['principal_or_first_maker'].map(merge_df['principal_or_first_maker'].value_counts())
        self.test['count_principal_or_first_maker'] = test['principal_or_first_maker'].map(merge_df['principal_or_first_maker'].value_counts())

class SubtitleBasefeature(Feature):
    def create_features(self):
        self.train['hw'] = train['size_h'] * train['size_w']
        self.test['hw'] = test['size_h'] * test['size_w']
        self.train['hwd'] = train['size_h'] * train['size_w'] * train['size_d']
        self.test['hwd'] = test['size_h'] * test['size_w'] * test['size_d']
        self.train['diff_hw'] = train['size_h'] - train['size_w']
        self.test['diff_hw'] = test['size_h'] - test['size_w']
        self.train['per_hw'] = train['size_h'] / train['size_w']
        self.test['per_hw'] = test['size_h'] / test['size_w']

class CopyrightholderLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['copyright_holder'].fillna(''))
        self.train['label_copyright_holder'] = features[:len(train)]
        self.test['label_copyright_holder'] = features[len(train):]

class CopyrightholderCountBasefeature(Feature):
    def create_features(self):
        self.train['count_copyright_holder'] = train['copyright_holder'].map(merge_df['copyright_holder'].value_counts())
        self.test['count_copyright_holder'] = test['copyright_holder'].map(merge_df['copyright_holder'].value_counts())

class MoretitleCount(Feature):
    def create_features(self):
        features = get_text_features_count(merge_df, 'more_title', 10, True)
        self.train = features[:len(train)]
        self.test = features[len(train):].reset_index(drop=True)

class MoretitleTfidf(Feature):
    def create_features(self):
        features = get_text_features_tfidf(merge_df, 'more_title', 10, True)
        self.train = features[:len(train)]
        self.test = features[len(train):].reset_index(drop=True)

class MoretitleBasefeature(Feature):
    def create_features(self):
        self.train['len_more_title'] = train['more_title'].fillna('').map(lambda x: len(x))
        self.test['len_more_title'] = test['more_title'].fillna('').map(lambda x: len(x))
        self.train['num_words_more_title'] = train['more_title'].fillna('').map(lambda x: len(x.split(' ')))
        self.test['num_words_more_title'] = test['more_title'].fillna('').map(lambda x: len(x.split(' ')))

class MoretitleLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['more_title'].fillna(''))
        self.train['label_more_title'] = features[:len(train)]
        self.test['label_more_title'] = features[len(train):]

class MoretitleCountBasefeature(Feature):
    def create_features(self):   
        self.train['count_more_title'] = train['more_title'].map(merge_df['more_title'].value_counts())
        self.test['count_more_title'] = test['more_title'].map(merge_df['more_title'].value_counts())

class AcquisitionmethodLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['acquisition_method'].fillna(''))
        self.train['label_acquisition_method'] = features[:len(train)]
        self.test['label_acquisition_method'] = features[len(train):]

class AcquisitionmethodCountBasefeature(Feature):
    def create_features(self):   
        self.train['count_acquisition_method'] = train['acquisition_method'].map(merge_df['acquisition_method'].value_counts())
        self.test['count_acquisition_method'] = test['acquisition_method'].map(merge_df['acquisition_method'].value_counts())

class AcquisitiondateBasefeature(Feature):
    def create_features(self):
        train_date = pd.to_datetime(train['acquisition_date'])
        test_date = pd.to_datetime(test['acquisition_date'])

        self.train['year_month_acquisition_date'] = train_date.map(lambda x: x.year+(x.month-1)/12)
        self.test['year_month_acquisition_date'] = test_date.map(lambda x: x.year+(x.month-1)/12)
        self.train['month_acquisition_date'] = train_date.map(lambda x: x.month)
        self.test['month_acquisition_date'] = test_date.map(lambda x: x.month)

class AcquisitioncreditlineTfidf(Feature):
    def create_features(self):
        features = get_text_features_tfidf(merge_df, 'acquisition_credit_line', 10, True)
        self.train = features[:len(train)]
        self.test = features[len(train):].reset_index(drop=True)

class AcquisitioncreditlineBasefeature(Feature):
    def create_features(self):
        self.train['len_acquisition_credit_line'] = train['acquisition_credit_line'].fillna('').map(lambda x: len(x))
        self.test['len_acquisition_credit_line'] = test['acquisition_credit_line'].fillna('').map(lambda x: len(x))
        self.train['num_words_acquisition_credit_line'] = train['acquisition_credit_line'].fillna('').map(lambda x: len(x.split(' ')))
        self.test['num_words_acquisition_credit_line'] = test['acquisition_credit_line'].fillna('').map(lambda x: len(x.split(' ')))


class AcquisitioncreditlineLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['acquisition_credit_line'].fillna(''))
        self.train['label_acquisition_credit_line'] = features[:len(train)]
        self.test['label_acquisition_credit_line'] = features[len(train):]

class AcquisitioncreditlineCountBasefeature(Feature):
    def create_features(self):
        self.train['count_acquisition_credit_line'] = train['acquisition_credit_line'].map(merge_df['acquisition_credit_line'].value_counts())
        self.test['count_acquisition_credit_line'] = test['acquisition_credit_line'].map(merge_df['acquisition_credit_line'].value_counts())

class DatingdateBasefeature(Feature):
    def create_features(self):
        self.train['dating_sorting_date'] = train['dating_sorting_date']
        self.train['dating_period'] = train['dating_period']
        self.train['dating_year_early'] = train['dating_year_early']
        self.train['dating_year_late'] = train['dating_year_late']
        self.train['dating_year_diff'] = train['dating_year_late'] - train['dating_year_early']

        self.test['dating_sorting_date'] = test['dating_sorting_date']
        self.test['dating_period'] = test['dating_period']
        self.test['dating_year_early'] = test['dating_year_early']
        self.test['dating_year_late'] = test['dating_year_late']
        self.test['dating_year_diff'] = test['dating_year_late'] - test['dating_year_early']

class MaterialBasefeature(Feature):
    def create_features(self):
        _df = pd.crosstab(material_df['object_id'], material_df['name'])
        svd = TruncatedSVD(n_components=10, random_state=42)
        features = pd.DataFrame(
            svd.fit_transform(_df),
            index = _df.index,
            columns = [f'svd_material_{i}' for i in range(10)])
        self.train = train[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.train['count_material'] = train['object_id'].map(material_df.groupby('object_id')['object_id'].count())
        self.test['count_material'] = test['object_id'].map(material_df.groupby('object_id')['object_id'].count())

class ObjectCollectionBasefeature(Feature):
    def create_features(self):
        _df = pd.crosstab(object_collection_df['object_id'], object_collection_df['name']).add_prefix('collect_')
        self.train = train[['object_id']].merge(_df, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(_df, on='object_id', how='left').drop(columns='object_id', axis=1)

class HistoricalpersonBasefeature(Feature):
    def create_features(self):
        _df = pd.crosstab(historical_person_df['object_id'], historical_person_df['name'])
        svd = TruncatedSVD(n_components=10, random_state=42)
        features = pd.DataFrame(
            svd.fit_transform(_df),
            index = _df.index,
            columns = [f'svd_historical_person_{i}' for i in range(10)])
        self.train = train[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.train['count_historical_person'] = train['object_id'].map(historical_person_df.groupby('object_id')['object_id'].count())
        self.test['count_historical_person'] = test['object_id'].map(historical_person_df.groupby('object_id')['object_id'].count())

class ProductionplaceBasefeature(Feature):
    def create_features(self):
        _df = pd.crosstab(production_place_df['object_id'], production_place_df['name'])
        svd = TruncatedSVD(n_components=10, random_state=42)
        features = pd.DataFrame(
            svd.fit_transform(_df),
            index = _df.index,
            columns = [f'svd_production_place_{i}' for i in range(10)])
        self.train = train[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.train['count_production_place'] = train['object_id'].map(production_place_df.groupby('object_id')['object_id'].count())
        self.test['count_production_place'] = test['object_id'].map(production_place_df.groupby('object_id')['object_id'].count())

class TechniqueBasefeature(Feature):
    def create_features(self):
        _df = pd.crosstab(technique_df['object_id'], technique_df['name'])
        svd = TruncatedSVD(n_components=10, random_state=42)
        features = pd.DataFrame(
            svd.fit_transform(_df),
            index = _df.index,
            columns = [f'svd_technique_{i}' for i in range(10)])
        self.train = train[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.train['count_technique'] = train['object_id'].map(technique_df.groupby('object_id')['object_id'].count())
        self.test['count_technique'] = test['object_id'].map(technique_df.groupby('object_id')['object_id'].count())

class PlaceofbirthLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['place_of_birth'].fillna(''))
        self.train['label_place_of_birth'] = features[:len(train)]
        self.test['label_place_of_birth'] = features[len(train):]

class PlaceofbirthCountBasefeature(Feature):
    def create_features(self):
        self.train['count_place_of_birth'] = train['place_of_birth'].map(merge_df['place_of_birth'].value_counts())
        self.test['count_place_of_birth'] = test['place_of_birth'].map(merge_df['place_of_birth'].value_counts())

class PlaceofdeathLabelBasefeature(Feature):
    def create_features(self):
        features = LabelEncoder().fit_transform(merge_df['place_of_death'].fillna(''))
        self.train['label_place_of_death'] = features[:len(train)]
        self.test['label_place_of_death'] = features[len(train):]

class PlaceofdeathCountBasefeature(Feature):
    def create_features(self):
        self.train['count_place_of_death'] = train['place_of_death'].map(merge_df['place_of_death'].value_counts())
        self.test['count_place_of_death'] = test['place_of_death'].map(merge_df['place_of_death'].value_counts())

class DatebirthdeathBasefeature(Feature):
    def create_features(self):
        self.train['tmp'] = np.zeros(len(train))
        self.test['tmp'] =  np.zeros(len(test))

        self.train.loc[train['date_of_birth'].notnull(),  'date_of_birth'] =\
             train.loc[train['date_of_birth'].notnull(),  'date_of_birth'].map(lambda x: int(x.split('-')[0]))
        self.train.loc[train['date_of_death'].notnull(),  'date_of_death'] =\
             train.loc[train['date_of_death'].notnull(),  'date_of_death'].map(lambda x: int(x.split('-')[0]))
        self.train['oldyear'] =  self.train['date_of_death'] - self.train['date_of_birth']
        self.test.loc[test['date_of_birth'].notnull(),  'date_of_birth'] =\
             test.loc[test['date_of_birth'].notnull(),  'date_of_birth'].map(lambda x: int(x.split('-')[0]))
        self.test.loc[test['date_of_death'].notnull(),  'date_of_death'] =\
             test.loc[test['date_of_death'].notnull(),  'date_of_death'].map(lambda x: int(x.split('-')[0]))
        self.test['oldyear'] =  self.test['date_of_death'] - self.test['date_of_birth']

        self.train = self.train.drop(columns=['tmp'], axis=1)
        self.test = self.test.drop(columns=['tmp'], axis=1)

class MakernameBasefeature(Feature):
    def create_features(self):
        _df = pd.crosstab(principal_maker_df['object_id'], principal_maker_df['maker_name'].fillna('nan'))
        svd = TruncatedSVD(n_components=10, random_state=42)
        features = pd.DataFrame(
            svd.fit_transform(_df),
            index = _df.index,
            columns = [f'svd_maker_name_{i}' for i in range(10)])
        self.train = train[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)

class QualificationBasefeature(Feature):
    def create_features(self):
        _df = pd.crosstab(principal_maker_df['object_id'], principal_maker_df['qualification'].fillna('nan'))
        svd = TruncatedSVD(n_components=10, random_state=42)
        features = pd.DataFrame(
            svd.fit_transform(_df),
            index = _df.index,
            columns = [f'svd_qualification_{i}' for i in range(10)])

        self.train = train[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)

class RolesBasefeature(Feature):
    def create_features(self):
        _df = pd.crosstab(principal_maker_df['object_id'], principal_maker_df['roles'].fillna('nan'))
        svd = TruncatedSVD(n_components=10, random_state=42)
        features = pd.DataFrame(
            svd.fit_transform(_df),
            index = _df.index,
            columns = [f'svd_roles_{i}' for i in range(10)])
        self.train = train[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)

class ProductionPlacesfeature(Feature):
    def create_features(self):
        _df = pd.crosstab(principal_maker_df['object_id'], principal_maker_df['productionPlaces'].fillna('nan'))
        svd = TruncatedSVD(n_components=10, random_state=42)
        features = pd.DataFrame(
            svd.fit_transform(_df),
            index = _df.index,
            columns = [f'svd_productionPlaces_{i}' for i in range(10)])
        self.train = train[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(features, on='object_id', how='left').drop(columns='object_id', axis=1)


class PrincipalMakerOccupatiofeature(Feature):
    def create_features(self):
        _df = pd.crosstab(principal_maker_occupatio_df['id'], principal_maker_occupatio_df['name']).add_prefix('occupatio_')
        self.train = train[['object_id']].merge(
            principal_maker_df.merge(_df, on='id', how='left')[['object_id'] + _df.columns.tolist()].groupby('object_id').sum(), on='object_id', how='left'
        ).drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(
            principal_maker_df.merge(_df, on='id', how='left')[['object_id'] + _df.columns.tolist()].groupby('object_id').sum(), on='object_id', how='left'
        ).drop(columns='object_id', axis=1)


class Color2vecDimensionDeletionfeature(Feature):
    def create_features(self):
        color2vec_path = Path('../../../features/colum2131/color2vec.pickle')
        if color2vec_path.exists():
            with open(color2vec_path, "rb") as fh:
                features = pickle5.load(fh)
            vectorizer = make_pipeline(
                make_union(
                    PCA(n_components=10, random_state=42),
                    n_jobs=1)
                )
            X = vectorizer.fit_transform(features).astype(np.float32)
            output_df = pd.DataFrame(X, columns=[
                f'color2vec_pca_{i}' for i in range(10)], 
                index=features.index
            )
            output_df['object_id'] = output_df.index
            self.train = train[['object_id']].merge(output_df, on='object_id', how='left').drop(columns='object_id', axis=1)
            self.test = test[['object_id']].merge(output_df, on='object_id', how='left').drop(columns='object_id', axis=1)
        else:
            print(f'{color2vec_path} no exists!')

class BertDimensionDeletionfeature(Feature):
    def create_features(self):
        self.train = pd.DataFrame(index=train.index)
        self.test = pd.DataFrame(index=test.index)

        for col in ['title', 'long_title', 'more_title', 'description', 'acquisition_credit_line']:
            features = pd.concat([
                pd.read_pickle(f'../../../features/colum2131/Bert_feature/bert_train_{col}.pickle'),
                pd.read_pickle(f'../../../features/colum2131/Bert_feature/bert_test_{col}.pickle')
            ]).reset_index(drop=True)
            vectorizer = make_pipeline(
                make_union(
                    PCA(n_components=10, random_state=42),
                    n_jobs=1)
                )
            X = vectorizer.fit_transform(features).astype(np.float32)
            output_df = pd.DataFrame(X, columns=[
                f'bert_pca_{col}_{i}' for i in range(10)],
                index=features.index
            )

            self.train = pd.concat([
                self.train,
                output_df[:len(train)]
            ], axis=1)

            self.test = pd.concat([
                self.test,
                output_df[len(train):].reset_index(drop=True)
            ], axis=1)

class UniversalsentenceencoderDimensionDeletionfeature(Feature):
    def create_features(self):
        self.train = pd.DataFrame(index=train.index)
        self.test = pd.DataFrame(index=test.index)

        for col in ['title', 'long_title', 'more_title', 'description', 'acquisition_credit_line']:
            features = pd.concat([
                pd.read_pickle(f'../../../features/colum2131/Universal_feature/train/{col}.pickle'),
                pd.read_pickle(f'../../../features/colum2131/Universal_feature/test/{col}.pickle')
            ]).reset_index(drop=True)

            vectorizer = make_pipeline(
                make_union(
                    PCA(n_components=10, random_state=42),
                    n_jobs=1)
                )

            X = vectorizer.fit_transform(features).astype(np.float32)
            output_df = pd.DataFrame(X, columns=[
                f'universalsentenceencoder_pca_{col}_{i}' for i in range(10)],
                index=features.index
            )

            self.train = pd.concat([
                self.train,
                output_df[:len(train)]
            ], axis=1)

            self.test = pd.concat([
                self.test,
                output_df[len(train):].reset_index(drop=True)
            ], axis=1)



class TargetencodingMaterialfeature(Feature):
    def create_features(self):
        _df = material_df.merge(train[['object_id', 'likes']], on='object_id', how='left')
        group = pd.concat([
            _df['object_id'],
            TargetEncoder(smoothing=0.1).fit_transform(_df['name'], np.log1p(_df['likes'])),
        ], axis=1).groupby('object_id').mean().rename(columns={'name':'targetencoding_material'})
        self.train = train[['object_id']].merge(group, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(group, on='object_id', how='left').drop(columns='object_id', axis=1)

class TargetencodingObjectcollectionfeature(Feature):
    def create_features(self):
        _df = object_collection_df.merge(train[['object_id', 'likes']], on='object_id', how='left')
        group = pd.concat([
            _df['object_id'],
            TargetEncoder(smoothing=0.1).fit_transform(_df['name'], np.log1p(_df['likes'])),
        ], axis=1).groupby('object_id').mean().rename(columns={'name':'targetencoding_object_collection'})
        self.train = train[['object_id']].merge(group, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(group, on='object_id', how='left').drop(columns='object_id', axis=1)

class TargetencodingTechniquefeature(Feature):
    def create_features(self):
        _df = technique_df.merge(train[['object_id', 'likes']], on='object_id', how='left')
        group = pd.concat([
            _df['object_id'],
            TargetEncoder(smoothing=0.1).fit_transform(_df['name'], np.log1p(_df['likes'])),
        ], axis=1).groupby('object_id').mean().rename(columns={'name':'targetencoding_technique'})
        self.train = train[['object_id']].merge(group, on='object_id', how='left').drop(columns='object_id', axis=1)
        self.test = test[['object_id']].merge(group, on='object_id', how='left').drop(columns='object_id', axis=1)

class TargetencodingDating_periodfeature(Feature):
    def create_features(self):
        _df = pd.concat([train, test]).reset_index(drop=True)
        _df = TargetEncoder(smoothing=0.1).fit_transform(
            _df['dating_period'].astype(object), np.log1p(_df['likes'])
        ).rename(columns={'dating_period':'targetencoding_dating_period'})
        self.train = _df[:len(train)]
        self.test = _df[len(train):].reset_index(drop=True)




if __name__ == '__main__':
    data_dir = Path('../../../data')
    train = pd.read_csv(data_dir.joinpath('train.csv'))
    test = pd.read_csv(data_dir.joinpath('test.csv'))
    sub = pd.read_csv(data_dir.joinpath('atmacup10__sample_submission.csv'))

    color_df = pd.read_csv(data_dir.joinpath('color.csv'))
    historical_person_df = pd.read_csv(data_dir.joinpath('historical_person.csv'))
    maker_df = pd.read_csv(data_dir.joinpath('maker.csv'))
    material_df = pd.read_csv(data_dir.joinpath('material.csv'))
    object_collection_df = pd.read_csv(data_dir.joinpath('object_collection.csv'))
    palette_df = pd.read_csv(data_dir.joinpath('palette.csv'))
    principal_maker_df = pd.read_csv(data_dir.joinpath('principal_maker.csv'))
    principal_maker_occupatio_df = pd.read_csv(data_dir.joinpath('principal_maker_occupation.csv'))
    production_place_df = pd.read_csv(data_dir.joinpath('production_place.csv'))
    technique_df = pd.read_csv(data_dir.joinpath('technique.csv'))

    train = creat_dataset(train)
    test = creat_dataset(test)

    merge_df = pd.concat([train, test]).reset_index(drop=True)
    print(merge_df.shape, train.shape, test.shape)

    class_list = [
        ArtseriesidLabelBasefeature(),
        ArtseriesidCountBasefeature(),
        TitleCount(),
        TitleTfidf(),
        TitleBasefeature(),
        TitleLabelBasefeature(),
        TitleCountBasefeature(),
        DescriptionCount(),
        DescriptionTfidf(),
        DescriptionBasefeature(),
        DescriptionLabelBasefeature(),
        DescriptionCountBasefeature(),
        LongtitleCount(),
        LongtitleTfidf(),
        LongtitleBasefeature(),
        LongtitleLabelBasefeature(),
        LongtitleCountBasefeature(),
        PrincipalmakerLabelBasefeature(),
        PrincipalmakerCountBasefeature(),
        PrincipalorfirstmakerLabelBasefeature(),
        PrincipalorfirstmakerCountBasefeature(),
        SubtitleBasefeature(),
        CopyrightholderLabelBasefeature(),
        CopyrightholderCountBasefeature(),
        MoretitleCount(),
        MoretitleTfidf(),
        MoretitleBasefeature(),
        MoretitleLabelBasefeature(),
        MoretitleCountBasefeature(),
        AcquisitioncreditlineTfidf(),
        AcquisitioncreditlineBasefeature(),
        AcquisitionmethodLabelBasefeature(),
        AcquisitionmethodCountBasefeature(),
        AcquisitiondateBasefeature(),
        AcquisitioncreditlineLabelBasefeature(),
        AcquisitioncreditlineCountBasefeature(),
        DatingdateBasefeature(),
        MaterialBasefeature(),
        ObjectCollectionBasefeature(),
        HistoricalpersonBasefeature(),
        ProductionplaceBasefeature(),
        TechniqueBasefeature(),
        PlaceofbirthLabelBasefeature(),
        PlaceofbirthCountBasefeature(),
        PlaceofdeathLabelBasefeature(),
        PlaceofdeathCountBasefeature(),
        DatebirthdeathBasefeature(),
        MakernameBasefeature(),
        QualificationBasefeature(),
        RolesBasefeature(),
        ProductionPlacesfeature(),
        PrincipalMakerOccupatiofeature(),
        Color2vecDimensionDeletionfeature(),
        BertDimensionDeletionfeature(),
        UniversalsentenceencoderDimensionDeletionfeature(),
        TargetencodingMaterialfeature(),
        TargetencodingObjectcollectionfeature(),
        TargetencodingTechniquefeature(),
        TargetencodingDating_periodfeature()
        ]

    generate_features(class_list, False)

    tmp = []
    for cal in class_list:
        tmp.append(cal.name)
    print(tmp)