import scipy.io
import numpy as np
import pandas as pd


if __name__ == '__main__':
    read_path = 'data/mat/RESCH/P05.mat'
    data = scipy.io.loadmat(read_path)


    cols=[]
    for i in data:
        if '__' not in i :
           cols.append(i)
    temp_df=pd.DataFrame(columns=cols)
    for i in data:
        if '__' not in i :
           temp_df[i]=(data[i]).ravel()

    print(temp_df)
    save_path = 'data/csv/RESCH/P05.csv'
    # for i in data:
    #     if '__' not in i and 'readme' not in i:
    #         np.savetxt((save_path), data[i], delimiter=',')

    temp_df.to_csv(save_path, index=False)


# plot average time to complete each task based on which task in order it is
# plot average time to complete task