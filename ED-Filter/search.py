import heapq
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
import numpy as np
import tensorflow as tf
from pyitlib import discrete_random_variable as drv
import time

def load_data():
    data = pd.read_excel(r'C:\Users\mnaseriparsa\Documents\FilterBoostPaper\EatingDisorder2Classified.xlsx')
    df = pd.DataFrame(data)  # , columns=['FromUser'])

    df = df.iloc[:, 1:29]
    return df

def compute_mutual_information():
    df = load_data()
    df_data_row = df.iloc[:, 0:27]
    df_data_row = df_data_row[1:2]
    #df_data_row = df_data_row.values.flatten()
    df_data_row = pd.DataFrame(df_data_row).transpose()
    category = pd.DataFrame(df.iloc[:, 27:28])
    category = category[1:2]
    a = drv.information_mutual(df_data_row,df_data_row, cartesian_product= True)
    return a


def reemovNestings(l):
    output = []
    for i in l:
        if type(i) == list:
            reemovNestings(i)
        else:
            output.append(i)
    return output

def baseline():
    df = load_data()

    df = df[10000:11500]

    df = df.iloc[:, 0:29]

def compute_upper_bound(mutual_information_score, size, maximum, remaining_size, swap, df, accuracy):
    sum = 0
    for k in range(remaining_size):
        sum = sum + k

    pair_sum = 0
    swap1 = swap.copy()
    swap1.pop(0)
    c = 0
    for i in range(len(swap)):
        for j in range(i + 1,len(swap)):
            if(i != j):
                pair_sum = pair_sum +  mutual_info_classif(np.array(df.loc[:, swap[i]]).reshape(-1,1), df.loc[:, swap[j]])
                #swap1.pop(0)
                c +=1

    remaining = sum - c
    max_remaining = (remaining * maximum  + mutual_information_score) + pair_sum #+ (size) * maximum
    max = ((sum + remaining_size) * maximum)
    x = (max_remaining - 0) / (max)
    #return x + 1 / 2
    upper_bound = ((x - math.log10(40) + 1) / math.log10(39)) + 1
    if upper_bound < accuracy:
        upper_bound = accuracy
    return upper_bound

def baseline():
    for iteration in range(1, 7):
        df = load_data()
        data_list = []

        feature_list = list(df.columns[0:27])

        importances = mutual_info_classif(df.iloc[:, 0:27], df.iloc[:, 27:28])

        threshold = 0
        top_element = []
        maximum = max(importances)
        importances_dict = {}
        for i in range(len(importances)):
            if importances[i] >= 0.015:
                importances_dict[feature_list[i]] = importances[i]

        #model = GaussianNB()
        model = MultinomialNB()

        df = df[0:30]
        start_time = time.time()
        for i in importances_dict.keys():
            # Train the model using the training sets
            model.fit(np.array(df.loc[:, i]).reshape(-1, 1), df.loc[:, 'category'])

            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(np.array(df.loc[:, i]).reshape(-1, 1),
                                                                df.loc[:, 'category'],
                                                                test_size=0.2,
                                                                random_state=109)  # 70% training and 30% test

            # Train the model using the training sets
            model.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = model.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            upper_bound = compute_upper_bound(importances_dict[i], 1, maximum, iteration, [i], df, accuracy)
            heapq.heappush(data_list, (tuple((-accuracy, upper_bound, [i], importances_dict[i]))))

        while len(data_list) != 0:
            top = heapq.heappop(data_list)
            if top[1] < threshold:
                break
            for feature in importances_dict.keys():
                if len(top[2]) >= iteration:
                    break
                if feature not in (top[2]):
                    swap = []
                    swap += top[2]
                    swap += [feature]

                    # Train the model using the training sets
                    model.fit(df.loc[:, swap], df.loc[:, 'category'])

                    # Split dataset into training set and test set
                    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, swap],
                                                                        df.loc[:, 'category'],
                                                                        test_size=0.2,
                                                                        random_state=109)  # 70% training and 30% test

                    # Train the model using the training sets
                    model.fit(X_train, y_train)

                    # Predict the response for test dataset
                    y_pred = model.predict(X_test)
                    accuracy = metrics.accuracy_score(y_test, y_pred)
                    sum = top[3] + importances_dict[feature]
                    if len(swap) == iteration:
                        upper_bound = accuracy
                    else:
                        upper_bound = compute_upper_bound(sum, len(swap), maximum, iteration, swap, df, accuracy)
                    heapq.heappush(data_list, (tuple((-accuracy, upper_bound, swap, sum))))
            if -top[0] > threshold:
                threshold = -top[0]
                top_element = top

        response_time = (time.time() - start_time)
        print("--- %s seconds ---" % response_time)
        print("test case---" + str(iteration))
        print(top_element)
        f = open("Results\\results30_baseline_v2_0.015.txt", "a")
        f.write("--- %s seconds ---" % response_time + "\n")
        f.write("test case---" + str(iteration) + "\n")
        f.write(str(top_element))
        f.write("\n")
        f.close()
    return top

def greedy_based():
    df = load_data()
    data_list = []

    feature_list = list(df.columns[0:27])

    importances = mutual_info_classif(df.iloc[:, 0:27], df.iloc[:, 27:28])

    threshold = 0
    maximum = max(importances)

    new_importances = []
    top_element = []
    importances_dict = {}
    for i in range(len(importances)):
        if importances[i] >= 0.01:
            importances_dict[feature_list[i]] = importances[i]

    df = df[0:1000]
    start_time = time.time()

    model = GaussianNB()
    for i in importances_dict.keys():
        # Train the model using the training sets
        model.fit(np.array(df.loc[:, i]).reshape(-1, 1), df.loc[:, 'category'])

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(np.array(df.loc[:, i]).reshape(-1, 1),
                                                            df.loc[:, 'category'],
                                                            test_size=0.2,
                                                            random_state=109)  # 70% training and 30% test

        # Train the model using the training sets
        model.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        upper_bound = compute_upper_bound(importances_dict[i], 1, maximum, len(importances_dict.keys()), [i] ,df, accuracy)
        heapq.heappush(data_list, (tuple((-accuracy, upper_bound, [i], importances_dict[i]))))

    counter = 0
    while len(data_list) != 0:
        top = heapq.heappop(data_list)
        counter += 1
        if counter > 500:
            break
        if top[1] < threshold:
            break
        for feature in importances_dict.keys():
            if feature not in (top[2]):

                swap = []
                swap+=top[2]
                swap+=[feature]
                #swap = reemovNestings(swap)

                # Train the model using the training sets
                model.fit(df.loc[:, swap], df.loc[:, 'category'])

                # Split dataset into training set and test set
                X_train, X_test, y_train, y_test = train_test_split(df.loc[:, swap],
                                                                    df.loc[:, 'category'],
                                                                    test_size=0.2,
                                                                    random_state=109)  # 70% training and 30% test

                # Train the model using the training sets
                model.fit(X_train, y_train)

                # Predict the response for test dataset
                y_pred = model.predict(X_test)
                accuracy = metrics.accuracy_score(y_test, y_pred)
                if  accuracy > - top[0]:
                    sum = top[3] + importances_dict[feature]
                    upper_bound = compute_upper_bound(sum, len(swap), maximum, len(importances_dict.keys()), swap, df, accuracy)
                    heapq.heappush(data_list, (tuple((-accuracy, upper_bound, swap, sum))))

        top_features = top[2]
        for feature in top_features:

            swap = list(set(top_features) - set([feature]))
            if len(swap) == 0:
                continue
            # Train the model using the training sets
            model.fit(df.loc[:, swap], df.loc[:, 'category'])

            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(df.loc[:, swap],
                                                                df.loc[:, 'category'],
                                                                test_size=0.2,
                                                                random_state=109)  # 70% training and 30% test

            # Train the model using the training sets
            model.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = model.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            if accuracy > - top[0]:
                sum = top[3] - importances_dict[feature]
                upper_bound = compute_upper_bound(sum, len(swap), maximum, len(importances_dict.keys()), swap, df, accuracy)
                heapq.heappush(data_list, (tuple((-accuracy, upper_bound, swap, sum))))
        if -top[0] > threshold:
            threshold = -top[0]
            top_element = top

    response_time = (time.time() - start_time)
    print("--- %s seconds ---" % response_time)
    #print("test case---" + str(iteration))
    print(top_element)
    f = open("Results\\results_Greedy_500_0.01.txt", "a")
    f.write("--- %s seconds ---" % response_time + "\n")
    #f.write("test case---" + str(iteration) + "\n")
    f.write(str(top_element))
    f.write("\n")
    f.close()

    return top

def hybrid_greedy_deep_learning():
    test_case = 0
    test_case_iteration = 100
    df = load_data()

    df_data_row = df.iloc[:, 0:29]
    #for iteration in range(0,11500,test_case_iteration):
    for iteration in range(1, 10):

        #df_data_row = df_data_row[200:300]
        df = load_data()
        df_data_row = df.iloc[:, 0:29]

        df_data_row = df_data_row[iteration:iteration + test_case_iteration]
        df_data_row = df_data_row.values.flatten()
        df_data_row = pd.DataFrame(df_data_row).transpose()

        model = tf.keras.models.load_model('saved_model/my_model')
        pred = np.argmax(model.predict(df_data_row))
        #pred = iteration

        data_list = []

        feature_list = list(df.columns[0:27])

        importances = mutual_info_classif(df.iloc[:, 0:27], df.iloc[:, 27:28])

        threshold = 0
        maximum = max(importances)

        new_importances = []
        top_element = []
        importances_dict = {}
        for i in range(len(importances)):
            if importances[i] >= 0.015:
                importances_dict[feature_list[i]] = importances[i]

        #df = df[test_case:iteration + test_case_iteration]
        df = df[0:100]

        start_time = time.time()
        #model = GaussianNB()
        model = MultinomialNB()
        for i in importances_dict.keys():
            # Train the model using the training sets
            model.fit(np.array(df.loc[:, i]).reshape(-1, 1), df.loc[:, 'category'])

            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(np.array(df.loc[:, i]).reshape(-1, 1),
                                                                df.loc[:, 'category'],
                                                                test_size=0.2,
                                                                random_state=109)  # 70% training and 30% test

            # Train the model using the training sets
            model.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = model.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            upper_bound = compute_upper_bound(importances_dict[i], 1, maximum, pred, [i], df, accuracy)
            heapq.heappush(data_list, (tuple((-accuracy, upper_bound, [i], importances_dict[i]))))

        counter = 0
        while len(data_list) != 0:
            top = heapq.heappop(data_list)
            counter += 1
            if counter > iteration * 200:
                if top[0] <= threshold:
                    break

            if top[1] < threshold:
                break
            #if counter > 200 and threshold > 0.8:
            #    break
            for feature in importances_dict.keys():
                if len(top[2]) >= pred:
                    break
                if feature not in (top[2]):

                    swap = []
                    swap+=top[2]
                    swap+=[feature]
                    #swap = reemovNestings(swap)

                    # Train the model using the training sets
                    model.fit(df.loc[:, swap], df.loc[:, 'category'])

                    # Split dataset into training set and test set
                    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, swap],
                                                                        df.loc[:, 'category'],
                                                                        test_size=0.2,
                                                                        random_state=109)  # 70% training and 30% test

                    # Train the model using the training sets
                    model.fit(X_train, y_train)

                    # Predict the response for test dataset
                    y_pred = model.predict(X_test)
                    accuracy = metrics.accuracy_score(y_test, y_pred)
                    if  accuracy > - top[0]:
                        sum = top[3] + importances_dict[feature]
                        if len(swap) == pred:
                            upper_bound = accuracy
                        else:
                            upper_bound = compute_upper_bound(sum, len(swap), maximum, pred, swap, df, accuracy)
                        heapq.heappush(data_list, (tuple((-accuracy, upper_bound, swap, sum))))

            top_features = top[2]
            for feature in top_features:
                #if len(top[2]) < pred:
                #    break
                swap = list(set(top_features) - set([feature]))
                if len(swap) == 0:
                    continue
                # Train the model using the training sets
                model.fit(df.loc[:, swap], df.loc[:, 'category'])

                # Split dataset into training set and test set
                X_train, X_test, y_train, y_test = train_test_split(df.loc[:, swap],
                                                                    df.loc[:, 'category'],
                                                                    test_size=0.2,
                                                                    random_state=109)  # 70% training and 30% test

                # Train the model using the training sets
                model.fit(X_train, y_train)

                # Predict the response for test dataset
                y_pred = model.predict(X_test)
                accuracy = metrics.accuracy_score(y_test, y_pred)
                if accuracy > - top[0]:
                    sum = top[3] - importances_dict[feature]
                    upper_bound = compute_upper_bound(sum, len(swap), maximum, pred, swap, df, accuracy)
                    heapq.heappush(data_list, (tuple((-accuracy, upper_bound, swap, sum))))
            if -top[0] > threshold:
                threshold = -top[0]
                top_element = top

        response_time = (time.time() - start_time)
        print("--- %s seconds ---" % response_time)
        print("test case---"+ str(iteration))
        print(top_element)
        f = open("Results\\results9000_0.035.txt", "a")
        f.write("--- %s seconds ---" % response_time + "\n")
        f.write("test case---"+ str(iteration) + "\n")
        f.write(str(top_element))
        f.write("\n")
        f.close()
    return top



if __name__ == '__main__':
    #baseline()
    greedy_based()
    #hybrid_greedy_deep_learning()
    #df = df[10000:11500]

    #df = df.iloc[:, 0:29]

