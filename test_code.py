#!/usr/bin/python3

import analysis as ana

def test():
    '''
    This is the function that runs test on all my functions.
    '''
    data_df = ana.loadAndCleanData(train_fname = 'train_codetest.csv', prop_fname = 'properties_codetest.csv')
    print(data_df.head())
    
    ana.summarizeValues(data_df)
    return

if __name__ == '__main__':
    test()
