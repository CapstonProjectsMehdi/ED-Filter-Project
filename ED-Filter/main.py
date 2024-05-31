from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xlsxwriter
import pandas as pd
import itertools
import numpy
import tensorflow as tf
import os

def read_data():
    data = pd.read_excel(r'C:\Users\mnaseriparsa\Documents\FilterBoostPaper\EatingDisorder2 - Copy.xlsx')
    df = pd.DataFrame(data)#, columns=['FromUser'])

    workbook = xlsxwriter.Workbook(r'C:\Users\mnaseriparsa\Documents\FilterBoostPaper\EatingDisorder2Classified.xlsx')
    worksheet = workbook.add_worksheet()

    row = 0
    col = 0
    worksheet.write(0, 0, 'UserName')
    worksheet.write(0, 1, 'RT')
    worksheet.write(0, 2, 'ExplicitEDTerms')
    worksheet.write(0, 3, 'AgainstED')
    worksheet.write(0, 4, 'ProAnaFamily')
    worksheet.write(0, 5, 'BodyAndBodyImage')
    worksheet.write(0, 6, 'BodyWeight')
    worksheet.write(0, 7, 'FoodAndMeals')
    worksheet.write(0, 8, 'EatOrAte')
    worksheet.write(0, 9, 'CaloricRestriction')
    worksheet.write(0, 10, 'Binge')
    worksheet.write(0, 11, 'ContemporaryBehavior')
    worksheet.write(0, 12, 'Exercise')
    worksheet.write(0, 13, 'Thinspo')
    worksheet.write(0, 14, 'Fitspo')
    worksheet.write(0, 15, 'Beauty')
    worksheet.write(0, 16, 'halloween')
    worksheet.write(0, 17, 'xmas')
    worksheet.write(0, 18, 'summer')
    worksheet.write(0, 19, 'winter')
    worksheet.write(0, 20, 'emo')
    worksheet.write(0, 21, 'BodyParts')
    worksheet.write(0, 22, 'Accessory')
    worksheet.write(0, 23, 'suicide')
    worksheet.write(0, 24, 'depressed')
    worksheet.write(0, 25, 'mentalhealth')
    worksheet.write(0, 26, 'domesticviolence')
    worksheet.write(0, 27, 'bullying')
    worksheet.write(0, 28, 'category')

    for index, row in df.iterrows():
        body_image = int(row['Unnamed: 5']) + int(row['Unnamed: 6']) + int(row['Unnamed: 21'])
        food = int(row['Unnamed: 7']) + int(row['Unnamed: 8']) + int(row['Unnamed: 9'])
        inspiration = int(row['Unnamed: 12']) + int(row['Unnamed: 13']) + int(row['Unnamed: 14']) + int(row['Unnamed: 15'])
        symptoms = int(row['Unnamed: 23']) + int(row['Unnamed: 24']) + int(row['Unnamed: 25']) + int(row['Unnamed: 26']) + int(row['Unnamed: 27'])

        category = 0
        maxv = max(body_image,food,inspiration,symptoms)
        if maxv == body_image:
            category = 0
        elif maxv == food:
            category = 1
        elif maxv == inspiration:
            category = 2
        else:
            category = 3


        worksheet.write(index + 1, col, row['Unnamed: 0'])
        worksheet.write(index + 1, col + 1, row['Unnamed: 1'])
        worksheet.write(index + 1, col + 2, row['Unnamed: 2'])
        worksheet.write(index + 1, col + 3, row['Unnamed: 3'])
        worksheet.write(index + 1, col + 4, row['Unnamed: 4'])
        worksheet.write(index + 1, col + 5, row['Unnamed: 5'])
        worksheet.write(index + 1, col + 6, row['Unnamed: 6'])
        worksheet.write(index + 1, col + 7, row['Unnamed: 7'])
        worksheet.write(index + 1, col + 8, row['Unnamed: 8'])
        worksheet.write(index + 1, col + 9, row['Unnamed: 9'])
        worksheet.write(index + 1, col + 10, row['Unnamed: 10'])
        worksheet.write(index + 1, col + 11, row['Unnamed: 11'])
        worksheet.write(index + 1, col + 12, row['Unnamed: 12'])
        worksheet.write(index + 1, col + 13, row['Unnamed: 13'])
        worksheet.write(index + 1, col + 14, row['Unnamed: 14'])
        worksheet.write(index + 1, col + 15, row['Unnamed: 15'])
        worksheet.write(index + 1, col + 16, row['Unnamed: 16'])
        worksheet.write(index + 1, col + 17, row['Unnamed: 17'])
        worksheet.write(index + 1, col + 18, row['Unnamed: 18'])
        worksheet.write(index + 1, col + 19, row['Unnamed: 19'])
        worksheet.write(index + 1, col + 20, row['Unnamed: 20'])
        worksheet.write(index + 1, col + 21, row['Unnamed: 21'])
        worksheet.write(index + 1, col + 22, row['Unnamed: 22'])
        worksheet.write(index + 1, col + 23, row['Unnamed: 23'])
        worksheet.write(index + 1, col + 24, row['Unnamed: 24'])
        worksheet.write(index + 1, col + 25, row['Unnamed: 25'])
        worksheet.write(index + 1, col + 26, row['Unnamed: 26'])
        worksheet.write(index + 1, col + 27, row['Unnamed: 27'])
        worksheet.write(index + 1, col + 28, category)

    workbook.close()

def prepare_training_model():
    df = pd.read_csv('train.csv', sep=',' )
    df_x = df.iloc[:, 0:2800]
    df_y = df.iloc[:, 2800:2801]

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0)

    #checkpoint_path = "training_1/cp.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dense(120, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(17, activation=tf.nn.softmax))

    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)

    model.save('saved_model/my_model')
    return model
    # predictions = model.predict(x_test)
    #
    # for i in range(0, len(x_test)):
    #     print(numpy.argmax(predictions[i]))

def run_model(model, data_row):
    df = data_row
    data_row = data_row.values.flatten()
    data_row = pd.DataFrame(data_row).transpose()
    pred = numpy.argmax(model.predict(data_row))

    importances = mutual_info_classif(df.iloc[:, 0:27], df.iloc[:, 27:28])
    indexlist = []
    index = 0
    for i in importances:
        if i >= 0.003:
            indexlist.append(index)
        index += 1

    columns = list(df.columns[indexlist])

    optionlist = list(itertools.combinations(columns, pred))

    max_element = []
    max_accuracy = 0
    for j in optionlist:

        if len(j) == 0:
            continue
        # Create a Gaussian Classifier
        model = GaussianNB()

        # Train the model using the training sets
        model.fit(df.loc[:, list(j)], df.loc[:, 'category'])

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(df.loc[:, list(j)], df.loc[:, 'category'],
                                                            test_size=0.2,
                                                            random_state=109)  # 70% training and 30% test

        # Train the model using the training sets
        model.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        if accuracy >= max_accuracy:
            max_element = j
            max_accuracy = accuracy
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print(j)
    print("max accuracy: ", max_accuracy)
    print("max element: ", max_element)


def load_data():
    data = pd.read_excel(r'C:\Users\mnaseriparsa\Documents\FilterBoostPaper\EatingDisorder2Classified.xlsx')
    df = pd.DataFrame(data)  # , columns=['FromUser'])

    df = df.iloc[:, 1:29]
    return df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #data.to_csv("train.csv", index_label=None, header=False, index=False, mode='a')
    #prepare_training_model()
    #data = pd.read_excel(r'C:\Users\mnaseriparsa\Documents\FilterBoostPaper\EatingDisorder2Classified.xlsx')
    #df = pd.DataFrame(data)  # , columns=['FromUser'])
    df = load_data()

    df = df[10000:11500]

    df = df.iloc[:, 0:29]

    model = prepare_training_model()
    start_index = 1500
    for row_index in range(1600, 1700, 100):
        data_row = df[start_index:row_index]
        run_model(model,data_row)
        start_index = row_index

    # start_index = 800
    # for row_index in range(900,10000,100):
    #
    #     row_data = df[start_index:row_index]
    #
    #     importances = mutual_info_classif(row_data.iloc[:, 0:27], row_data.iloc[:, 27:28])
    #     indexlist = []
    #     index = 0
    #     for i in importances:
    #         if i >= 0.003:
    #             indexlist.append(index)
    #         index += 1
    #
    #     #columns = list(df.columns[1:28])
    #
    #     columns = list(row_data.columns[indexlist])
    #     optionlist = list(itertools.chain.from_iterable(itertools.combinations(columns, r) for r in range(len(columns) + 1)))
    #
    #     max_element = []
    #     max_accuracy = 0
    #     for j in optionlist:
    #
    #         if len(j) == 0:
    #             continue
    #         row_data.loc[:, list(j)]
    #         # Create a Gaussian Classifier
    #         model = GaussianNB()
    #
    #         # Train the model using the training sets
    #         model.fit(row_data.loc[:, list(j)], row_data.loc[:, 'category'])
    #
    #         # Split dataset into training set and test set
    #         X_train, X_test, y_train, y_test = train_test_split(row_data.loc[:, list(j)], row_data.loc[:, 'category'], test_size=0.2,
    #                                                         random_state=109)  # 70% training and 30% test
    #
    #         # Train the model using the training sets
    #         model.fit(X_train, y_train)
    #
    #         # Predict the response for test dataset
    #         y_pred = model.predict(X_test)
    #         accuracy = metrics.accuracy_score(y_test, y_pred)
    #         if accuracy >= max_accuracy:
    #             max_element = j
    #             max_accuracy = accuracy
    #         print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    #         print(j)
    #
    #     print("Max Accuracy:", max_accuracy)
    #     print("Max Element:", max_element)
    #     #dat1 = pd.DataFrame({'Cardinality': [len(max_element) for x in range(100)]})
    #     #dat1 = pd.DataFrame({'Cardinality': len(max_element)})
    #     row_data = row_data.values.flatten()
    #     row_data = numpy.append(row_data, len(max_element))
    #     row_data = pd.DataFrame(row_data).transpose()
    #
    #     row_data.to_csv("train.csv", index_label=None, header=False, index=False, mode='a')
    #
    #     start_index = row_index


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
