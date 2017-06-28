#!/usr/bin/python3

#==============================================================
# This document contains all the methods used to process and
# analyze the Zillow housing data. Note that it is assumed this
# is run in a local directory with all the data files present.
#==============================================================
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from math import floor

#==============================================================
# These were defined to make sample files for testing. Sadly,
# they failed. Trying to make a sample file that doesn't create
# special glitches proved very hard.
#==============================================================
def loadLinesAndGetRand(fname, n_rand):
    '''
    Loads the lines as a list of strings from a given file and gives a random
    list of numbers that index that list. Also separates out the header.
    '''
    f = open(fname, 'r')
    headline = f.readline()
    line_list = f.readlines()
    f.close()
    
    rand_list = np.random.choice(range(0, len(line_list)), n_rand)
    
    map_dict = {re.match('^(\d+)', line).group():i for i, line in enumerate(line_list)}
    
    return headline, line_list, rand_list, map_dict

def createTestFiles(n_train, n_prop):
    '''
    Creates a new set of test files used to test the functionality of this code.
    '''
    tr_test_str, tr_line_list, tr_rand_list, tr_map_dict = loadLinesAndGetRand('train_2016.csv', n_train)
    pr_test_str, pr_line_list, pr_rand_list, pr_map_dict = loadLinesAndGetRand('properties_2016.csv', n_prop)
    
    for i_pr, line in enumerate(pr_line_list):
        id_num = re.match('^(\d+)', line).group()
        if id_num in tr_map_dict.keys():
            i_tr = tr_map_dict[id_num]
            if i_pr in pr_rand_list or i_tr in tr_rand_list:
                tr_test_str += tr_line_list[i_tr]
                pr_test_str += line
    
    tr_test_file = open('train_codetest.csv', 'w')
    tr_test_file.write(tr_test_str)
    tr_test_file.close()
    
    pr_test_file = open('properties_codetest.csv', 'w')
    pr_test_file.write(pr_test_str)
    pr_test_file.close()
    
    return

#===============================================================
# Here begin the actual useful functions and classes.
#===============================================================
def loadData(train_fname = 'train_2016.csv', prop_fname = 'properties_2016.csv'):
    '''
    Load the properties and training data and merge them (inner) into
    a single dataframe.
    
    Inputs:
    train_fname     -- the name of the csv file containing the training data.
                       Default: `train_2016.csv`
    prop_fname      -- the name of the csv file containing the properties data.
                       Default: `properties_2016.csv`
    
    Returns:
    data_df         -- the pandas dataframe with the cleaned data
    
    '''
    # Get the data_key_dict values.
    #key_dict = (pd.read_excel('zillow_data_dictionary.xlsx', sheetname = 'Data Dictionary')
    #                   .set_index('Feature')
    #                   .to_dict()['Description'])
    
    # Read in the raw csv
    raw_props_df = pd.read_csv(prop_fname).set_index('parcelid')
    
    # Read in the training data
    raw_train_df = pd.read_csv(train_fname).set_index('parcelid')
    # key_dict.update(zip(raw_train_df.columns, 
    #        ['The difference: log(Zestimate) - log(Actual value), to be predicted', 
    #         'The date of the transaction to which this is compaired']))
    
    df = pd.merge(raw_props_df, raw_train_df, how='inner', left_index=True, right_index=True)
    
    # Remove rows that are duplicates.
    df = df.reset_index()
    df = df.drop('parcelid', axis=1)
    df = df.drop_duplicates()
    
    return df
        

class Cleaner(object):
    '''
    This object is used to clean the data sets in a way that does not allow data leaks.
    
    You need to clean the training data first, and then clean the test data. If you need
    to re-prime, call cleanData with prime=True.
    '''
    def __init__(self):
        self.primed = False
        self.col_dists = {}
        self.col_drops = []
        self.col_idx_maps = {}
        self.scaler = MinMaxScaler() 
        return
    
    def __processDate(self, date_val):
        '''
        Quick function to convert a date str. into a float.
        '''
        ret = date_val
        if isinstance(date_val, str):
            d = datetime.strptime(date_val, '%Y-%m-%d')
            ret = d#(d - datetime(2016, 1, 1)).days/365
        return ret 
    
    def __breakDate(self, row_srs):
        '''
        Break up the date into year, month, and day, as separate columns
        
        Note: This is applied to rows, not columns, so do not set axis=1.
        '''
        if 'transactiondate' in row_srs:
            date_val = row_srs['transactiondate']
            base = 'transaction'
            if isinstance(date_val, str):
                # Convert the string to a datetime object.
                d = datetime.strptime(date_val, '%Y-%m-%d')
                
                # Enter the year and month
                row_srs[base + 'year'] = d.year
                row_srs[base + 'month'] = d.month
                
                # I make the days uniformly scaled in each month.
                days_in_month = (datetime(d.year + floor(d.month/12), d.month%12 + 1, 1) - timedelta(days=1)).day
                row_srs[base + 'day'] = d.day/days_in_month
                
                # Remove the original item.
                row_srs = row_srs.drop('transactiondate')
        
        return row_srs
    
    def __fixTypes(self, col_srs):
        '''
        Quick private function that makes it so column dtypes are accurate.
        '''
        print("Converting type", col_srs.dtype, "to true type", col_srs.unique().dtype)
        return col_srs.astype(col_srs.unique().dtype)
    
    def __convSqFt(self, col_srs):
        '''
        Quick private function to take the sqare root of any squared
        measurement
        '''
        sqft_list = ['sqft', 'squarefeet']
        name = col_srs.name
        
        new_name = ''
        for sqft in sqft_list:
            if sqft in name:
                print('Fixing squares, replacing', name)
                if 'size' not in name:
                    new_name = name.replace(sqft, 'size')
                else:
                    new_name = name.replace(sqft, '')
                
                col_srs.name = new_name
                col_srs = col_srs.apply(np.sqrt)
                break
        
        return col_srs
    
    def __convertIntsToInts(self, col_srs):
        '''
        Quick private function that converts ints currently represented as
        floats to ints.
        '''
        if col_srs.dtype == 'float64':
            float_locs = np.not_equal(np.mod(col_srs[col_srs.notnull()], 1), 0)
            thereIsAFloat = float_locs.any()
            if not thereIsAFloat:
                print("Converting", col_srs.name, "to int")
                col_srs = col_srs.astype('int')
            #else:
            #    print("Found floats:", col_srs[float_locs].values[0])
        
        return col_srs
    
    def __primeCol(self, col_srs):
        '''
        Quick private function that determines how to clean the data from
        the training set so that there is no data leakage. 
        '''
        #print("Priming", col_srs.dtype)
        col_name = col_srs.name
        
        # Get the dist of values that aren't null to be used filling
        # the null values.
        self.col_dists[col_name] = col_srs[col_srs.notnull()]
        
        
        # Look for columns with no information and columns that are populated
        # entirely by strings. NOTE: There is the potential for a bug if a label
        # exists in test but not training. This should only be a rare corner case,
        # though.
        unq = col_srs[col_srs.notnull()].unique()
        if len(unq) <= 1:
            self.col_drops.append(col_name)
        
        other_cols = ['propertylandusetypeid', 'propertycountylandusecode']
        if col_srs.dtype == 'object' or col_srs.name in other_cols:
            self.col_idx_maps[col_name] = {val:idx for idx,val in enumerate(unq)}
        
        return col_srs
    
    def __subRandForNaN(self, col_srs):
        '''
        Quick private function to substitute a random sample of values for NaN values
        '''
        if col_srs.isnull().any():
            print("Subbing rands for NaN's for", col_srs.name)
            dist = self.col_dists[col_srs.name]
            try:
                col_srs[col_srs.isnull()] = dist.sample(col_srs.isnull().sum(), replace=True).values
            except ValueError:
                print(col_srs.index.duplicated().any())
                raise
        
        return col_srs
    
    def __subIntsForStrings(self, col_srs):
        '''
        Quick private function to replace string ID's with integers using the mapping
        defined in __primeCol
        '''
        if col_srs.name in self.col_idx_maps.keys():
            print("Changing {} labels to int labesl for".format(col_srs.dtype), col_srs.name)
            col_srs = col_srs.apply(lambda val: self.col_idx_maps[col_srs.name][val])
        return col_srs
       
    
    def __scaleData(self, col_srs):
        '''
        Quick private function to scale the floating point data of relevant columns
        ''' 
        if col_srs.name in self.col_scalers:
            print("Scaling float data for", col_srs.name)
            col_srs = self.col_scalers[col_srs.name].transform(col_srs)
        return col_srs
        
    
    def cleanData(self, X, y, prime = False):
        '''
        This method makes all the procedures for cleaning the data.
        
        Inputs:
        X, y -- the dataframe to be cleaned.
        prime -- bool, default False: if True, then even if already primed
                with training data, prime again. Otherwise use what we got
                from the training data.
        
        Outputs:
        X, y -- the modfied dataframe. Note that the dataframe is not changed
                   in place.
        '''
        # Merge the inputs and outputs for the purposes of cleaning
        df = X.copy()
        df['outputs'] = y.copy()
        
        init_ncols = len(df.columns)
        
        df = df.apply(self.__fixTypes).apply(self.__convSqFt)
        
        print('Breaking up the years')
        df = df.apply(self.__breakDate, axis=1)
        print('Done breaking up the years')
        
        if not self.primed or prime:
            print('Priming')
            df = df.apply(self.__primeCol)
        
        df = df.apply(self.__subRandForNaN)
        df = df.apply(self.__convertIntsToInts)
        df = df.drop(self.col_drops, axis=1, errors='ignore')
        df = df.apply(self.__subIntsForStrings)
        
        if not self.primed or prime:
            #self.scaler.fit(df)
            self.primed = True
        
        #print("Scaling the data")
        #df.loc[:] = self.scaler.transform(df)
        
        return df.drop('outputs', axis=1), df['outputs']

def summarizeValues(data_df):
    '''
    Print out some summaries of the types of values present for each
    column.
    '''
    fmt = '{label:.<30}: len {len:<5} dtype: {dtype} min: {min} mean: {mean} max: {max} \n {unique}\n'
    prop_dict = {}
    for col_key in data_df.columns:
        unq = data_df[col_key].unique()
        prop_dict.update(zip(['label', 'len', 'dtype'], [col_key, len(unq), unq.dtype]))
        if len(unq) > 15:
            prop_dict['unique'] = str(unq[:15]) + ' etc'
        else:
            prop_dict['unique'] = str(unq)
        
        mmm_keys = ['min', 'mean', 'max']
        if unq.dtype == 'float64' or unq.dtype == 'int64':
            prop_dict.update(zip(mmm_keys,
                            [np.nanmin(unq), np.nanmean(unq), np.nanmax(unq)]))
        else:
            prop_dict.update(zip(mmm_keys, 3*['N/A']))
        
        print(fmt.format(**prop_dict))
    
    return

from sklearn.model_selection import train_test_split
def splitAndClean(df, test_size = 0.25):
    '''
    Quick script to clean split and clean the data. The input should be the 
    DataFrame that is output from loadData.
    '''
    X = df.drop('logerror', axis=1)
    y = df['logerror']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    c = Cleaner()
    
    X_train, y_train = c.cleanData(X_train, y_train)
    X_test, y_test = c.cleanData(X_test, y_test)
    
    return X_train, X_test, y_train, y_test

#====================================================================
# Here begin some plotting utilities.
#====================================================================

def plotMap(df, color_col = None):
    '''
    Quick and easy function to plot themap with an option of coloring
    it with data from another column.
    '''
    coloring = color_col is not None
    kwargs = dict(x = 'longitude', y = 'latitude', kind='scatter', s = 1, alpha = 0.01)
    
    if coloring:
        kwargs.update(dict(c = color_col, colorbar = coloring, colormap = 'Blues', alpha = 1))
    
    return df.plot(**kwargs)
