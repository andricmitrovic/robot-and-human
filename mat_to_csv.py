import scipy.io
import pandas as pd
import os


if __name__ == '__main__':
    rescheduling = False

    if rescheduling:
        subfolder = 'RESCH'
        idxs = [1, 2, 3, 4, 6, 7] # 5
    else:
        subfolder = 'NO_RESCH'
        idxs = [1, 2, 3] # 4, 5, 6
    # Create output dir for csv files
    dir_path = f"./data/csv/{subfolder}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Convert mat to csv
    for i in idxs:            # todo: some error generating csv for operator 5
        read_path = f'data/mat/{subfolder}/P0' + str(i) + '.mat'
        save_path = f'data/csv/{subfolder}/P0' + str(i) + '.csv'
        data = scipy.io.loadmat(read_path)

        cols=[]
        for i in data:
            if '__' not in i :
               cols.append(i)
        temp_df=pd.DataFrame(columns=cols)
        for i in data:
            if '__' not in i :
               temp_df[i]=(data[i]).ravel()

        temp_df.to_csv(save_path, index=False)
