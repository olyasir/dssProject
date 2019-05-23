import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CRYPTIC_PLATES = ['4Q249', '4Q250', '4Q298', '4Q317', '4Q324d-i', '4Q324d','4Q324', '4Q313', '11Q23']
BEST_COND = ['4Q298', '4Q317', '4Q249' ]

DATA_PATH ='data/Fragment on plate to DJD 27042017.xlsx'

def allPlates():
    data = pd.read_excel(DATA_PATH, dtype={'Plate number- IAA inventory ':str})
    crypticData = data[ data['Manuscript number '].isin(BEST_COND)]
    #saveCryptic(crypticData)
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    sns.countplot(x='Manuscript number ', hue='Plate number- IAA inventory ', data=crypticData, ax=axes[1])
    plt.show()

    existing = getExistingAllPlates()

    new_df = pd.merge(crypticData, existing, how='left',
                      left_on=['Plate number- IAA inventory ', 'Fragment number (on IAA plate)'],
                      right_on=['Plate number- IAA inventory ', 'Fragment number (on IAA plate)'])
    new_df = new_df.fillna(False)
    saveCryptic( new_df, 'crypticExist')
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    sns.countplot(x='Plate number- IAA inventory ', hue='exist', data=new_df, ax=axes[1])
    #sns.countplot(x='Plate number- IAA inventory ', data=new_df, ax=axes[1])
    plt.show()
    print("")




def getExistingAllPlates():
    import os
    plates = []
    fragments =[]
    for f in os.listdir('data/adielNoBg'):
        temp = f.split('-')
        plate = temp[0]
        plate = int(plate[1:])
        fragment = temp[1]
        fragment = int(fragment[2:])
        plates.append(str(-plate))
        fragments.append(fragment)
    return pd.DataFrame( { 'Plate number- IAA inventory ': plates, 'Fragment number (on IAA plate)':fragments, 'exist': [ True]*len(fragments) } ).drop_duplicates()






def saveCryptic(crypticData, name):
    writer = pd.ExcelWriter('data/{}.xlsx'.format(name))
    crypticData.to_excel(writer, 'Sheet1')
    writer.save()
    print("")


if __name__ =='__main__':
    allPlates()