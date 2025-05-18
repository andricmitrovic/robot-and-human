import scipy.io
import pandas as pd
import os


if __name__ == '__main__':
    rescheduling = False

    for mode in ['RESCH', 'NO_RESCH']:

        # Todo: Some corrupted .mat files
        if mode == 'RESCH':
            idxs = [1, 2, 3, 4, 6, 7] # 5
        else:
            idxs = [1, 2, 3] # 4, 5, 6

        # Create output dir for csv files
        dir_path = f"./data/csv/{mode}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Convert mat to csv
        for i in idxs:
            read_path = f'data/mat/{mode}/P0' + str(i) + '.mat'
            save_path = f'data/csv/{mode}/P0' + str(i) + '.csv'
            data = scipy.io.loadmat(read_path)

            cols=[]
            for i in data:
                if '__' not in i :
                   cols.append(i)
            df=pd.DataFrame(columns=cols)
            for i in data:
                if '__' not in i :
                   df[i]=(data[i]).ravel()

            df.to_csv(save_path, index=False)
