import pandas as pd
import sklearn.datasets as datasets

answer = input('Would you like to load the breast cancer dataset from sklearn? Enter yes or no! ').lower()

if answer == 'yes':
    data = datasets.load_breast_cancer()
    print(f'Good job. You just loaded the {data.filename} file!\n')

    while True:
        answer = input('To make a dataframe enter yes, else no! ').lower()
        if answer == 'no':
            print('Understood. Bye!')
            break
        elif answer == 'yes':
            df = pd.DataFrame(data.data, columns=data.feature_names)
            print(f'You have just transformed {data.filename} into a pandas DataFrame.\n')
            
            while True:
                options = ["info", "desc", "both", "none"]
                answer = input(f'Please pick a valid option from {options}! ').lower()
                if answer == "info":
                    print(df.info()); break
                elif answer == "desc":
                    print('-'*50,'\n',df.describe()); break
                elif answer == "both": 
                    print(df.info(),'\n',df.describe()); break
                elif answer == "none": break
                print(f'{answer} is not a valid option.'); continue

            dataframe = df; break
else:
    print('Understood, good bye!')