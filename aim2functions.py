import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 

from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore")
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def get_sentence_word_count(text_list):
    sent_count = 0
    word_count = 0
    vocab = {}
    for text in text_list:
        sentences=sent_tokenize(str(text).lower())
        sent_count = sent_count + len(sentences)
        for sentence in sentences:
            words=word_tokenize(sentence)
            for word in words:
                if(word in vocab.keys()):
                    vocab[word] = vocab[word] +1
                else:
                    vocab[word] =1 
    word_count = len(vocab.keys())
    return sent_count,word_count

def clean_text(text): 
    text = text.translate(str.maketrans('', '', string.punctuation))
    text1 = ''.join([w for w in text if not w.isdigit()]) 
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    #BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    
    text2 = text1.lower()
    text2 = REPLACE_BY_SPACE_RE.sub('', text2) # replace REPLACE_BY_SPACE_RE symbols by space in text
    #text2 = BAD_SYMBOLS_RE.sub('', text2)
    return text2

def lemmatize_text(text):
    wordlist=[]
    lemmatizer = WordNetLemmatizer() 
    sentences=sent_tokenize(text)
    
    intial_sentences= sentences[0:1]
    final_sentences = sentences[len(sentences)-2: len(sentences)-1]
    
    for sentence in intial_sentences:
        words=word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))
    for sentence in final_sentences:
        words=word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))       
    return ' '.join(wordlist) 

# Define a function to perform chi-square test and select top k features
def chi2_feature_selection(X, y, k):
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = X[:, selected_indices]
    
    # Convert selected features to binary values
    selected_features[selected_features > 0] = 1
    
    # Compute contingency table and p-values
    contingency_table = np.vstack([selected_features[y == label].sum(axis=0) for label in np.unique(y)])
    contingency_table += 1e-10  # Add a small constant to avoid zero elements
    _, p_values, _, _ = chi2_contingency(contingency_table)
    
    return selected_indices, p_values

# Define a function to print classification metrics
def print_classification_metrics(y_true, y_pred, target_names):
    print('Accuracy: {:.3f}'.format(accuracy_score(y_true, y_pred)))
    print('Precision: {:.3f}'.format(precision_score(y_true, y_pred, average='weighted')))
    print('Recall: {:.3f}'.format(recall_score(y_true, y_pred, average='weighted')))
    print('F1-score: {:.3f}'.format(f1_score(y_true, y_pred, average='weighted')))
    print('Classification Report:\n', classification_report(y_true, y_pred, target_names=target_names))

# define a function to clean and lemmatize text using nltk
def clean_and_lemmatize_text(text):
    # tokenize the text into words
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    
    # remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    
    # lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # join the words back into a string
    cleaned_text = ' '.join(words)
    
    return cleaned_text
    
def preprocess_text(df):
    '''
    :param df: dataframe contains the message, label, and code
    :return tfidf: matrix, 
            data:Dataframe processed
    '''
    import nltk

    df = df[df['Message'].notna()]
    replacements = {'Auto': 'A', 'Patient': 'P', 'Clinician': 'C'}

    # replace the values in the specified column
    df['Code'] = df['Code'].replace(replacements)

    sent_count,word_count= get_sentence_word_count(df['Message'].tolist())
    print("Number of sentences in transcriptions column: "+ str(sent_count))
    print("Number of unique words in transcriptions column: "+str(word_count))


    data_categories  = df.groupby(df['Label'])
    i = 1
    print('===========Original Categories =======================')
    for catName,dataCategory in data_categories:
        print('Cat:'+str(i)+' '+catName + ' : '+ str(len(dataCategory)) )
        i = i+1
    print('==================================')

    filtered_data_categories = data_categories.filter(lambda x:x.shape[0] > 10)
    final_data_categories = filtered_data_categories.groupby(filtered_data_categories['Label'])
    i=1
    print('============Reduced Categories ======================')
    for catName,dataCategory in final_data_categories:
        print('Cat:'+str(i)+' '+catName + ' : '+ str(len(dataCategory)) )
        i = i+1

    print('============ Reduced Categories ======================')

    # Group the data by Label and Code and count the number of messages in each group
    grouped_data = df.groupby(['Label', 'Code'])['Message'].count()

    # Convert the grouped data to a DataFrame
    grouped_df = grouped_data.to_frame().reset_index()

    # Create a pivot table to reshape the data for plotting
    pivot_df = grouped_df.pivot(index='Label', columns='Code', values='Message')

    # Create a bar plot of the data
    pivot_df.plot(kind='bar', stacked=True)
    plt.ylabel('Count')
    plt.title('Messages by Label and Code')
    plt.show()

    # define the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # apply the clean and lemmatize function to the 'Message' column
    data = df.copy()
    data['Message'] = data['Message'].apply(clean_and_lemmatize_text)

    # use the tf-idf vectorizer with the cleaned and lemmatized text
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english',ngram_range=(1,3), max_df=0.75, use_idf=True, smooth_idf=True, max_features=1000)
    tfIdfMat  = vectorizer.fit_transform(data['Message'].tolist() )
    feature_names = sorted(vectorizer.get_feature_names_out())

    import gc
    gc.collect()
    tfIdfMatrix = np.asarray(tfIdfMat.todense())
    labels = data['Label'].tolist()
    tsne_results = TSNE(n_components=2,init='random',random_state=0, perplexity=40).fit_transform(tfIdfMatrix)
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=labels,
        legend="full",
        alpha=1
    )

    gc.collect()

    labels = data['Label'].tolist()
    category_list = data.Label.unique()
    X_train, X_test, y_train, y_test = train_test_split(tfIdfMatrix, labels, stratify=labels,random_state=1)
    print('Train_Set_Size:'+str(X_train.shape))
    print('Test_Set_Size:'+str(X_test.shape)) 

    plt.figure(figsize=(8,8))

    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=labels,
        legend="full",
        alpha=1
    )
    for i, txt in enumerate(data['Code']):
        plt.annotate(txt, (tsne_results[:,0][i], tsne_results[:,1][i]))
    plt.show()

    plt.figure(figsize=(8,8))

    codes = data['Code'].tolist()
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=codes,
        legend="full",
        alpha=1
    )
    plt.show()

    return tfIdfMatrix, data

def perform_binary_logistic_regression(df, top_k=50):
   # Define binary classification tasks
    binary_tasks = {
        'Info-giving vs. Non-Info-giving': ['Info Giving', 'Non-Info-giving'],
        'Info-seeking vs. Non-info-seeking': ['Info Seeking', 'Non-info-seeking'],
        'Emotion vs. non-emotion': ['Emotion', 'Non-emotion'],
        'Partnership vs. non-partnership': ['Partnership', 'Non-partnership']
    }

    # Perform binary classification tasks
    selected_features = []
    for task_name, labels in binary_tasks.items():
        print('Performing binary classification for task:', task_name)
        X = df['Message'].values
        y = df['Label'].apply(lambda x: labels[0] if x == labels[0] else labels[1]).values

        # Convert text data to TF-IDF features
        vectorizer =   TfidfVectorizer(analyzer='word', stop_words='english')
        X = vectorizer.fit_transform(X)

        # Perform feature selection using chi-square test
        selected_indices, p_values = chi2_feature_selection(X, y, top_k)
        selected_features += [vectorizer.get_feature_names()[i] for i in selected_indices]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

        # Train a logistic regression model
        lr = LogisticRegression(max_iter=1000)
        param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
        clf = GridSearchCV(lr, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train[:, selected_indices], y_train)

        # Evaluate the model on the testing set
        y_pred = clf.predict(X_test[:, selected_indices])
        print_classification_metrics(y_test, y_pred, target_names=labels)
        print('Selected features:', ', '.join([vectorizer.get_feature_names()[i] for i in selected_indices]))
        print('')

    # Perform multi-class classification using selected features
    top_k *= 4  # Increase the number of top features to include all selected features from binary classification tasks
    selected_features = list(set(selected_features))[:top_k]
    print('Performing multi-class classification using top {} features:'.format(top_k))
    X = df['Message'].values
    y = df['Label'].values

    # Convert text data to TF-IDF features
    vectorizer =  TfidfVectorizer(analyzer='word',vocabulary=selected_features, stop_words='english')
    X = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Train a logistic regression model
    lr = LogisticRegression(max_iter=1000)
    param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
    clf = GridSearchCV(lr, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = clf.predict(X_test)

    # Get the unique label names in the order of the classes
    label_list = clf.classes_

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=label_list)

    # Create a heatmap of the confusion matrix with label names
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Multi-class Classification')
    plt.show()

    # Print classification report
    print('Classification Report:')
    print_classification_metrics(y_test, y_pred, target_names=label_list)
    print('Top 200 features:')
    for i, feature in enumerate(feature_names[:200]):
        print(i+1, feature)

from sklearn.tree import DecisionTreeClassifier

def perform_binary_decision_tree_classification(df, top_k=50):
    # Define binary classification tasks
    binary_tasks = {
        'Info-giving vs. Non-Info-giving': ['Info Giving', 'Non-Info-giving'],
        'Info-seeking vs. Non-info-seeking': ['Info Seeking', 'Non-info-seeking'],
        'Emotion vs. non-emotion': ['Emotion', 'Non-emotion'],
        'Partnership vs. non-partnership': ['Partnership', 'Non-partnership']
    }

    # Perform binary classification tasks
    selected_features = []
    for task_name, labels in binary_tasks.items():
        print('Performing binary classification for task:', task_name)
        X = df['Message'].values
        y = df['Label'].apply(lambda x: labels[0] if x == labels[0] else labels[1]).values

        # Convert text data to TF-IDF features
        vectorizer =   TfidfVectorizer(analyzer='word', stop_words='english')
        X = vectorizer.fit_transform(X)

        # Perform feature selection using chi-square test
        selected_indices, p_values = chi2_feature_selection(X, y, top_k)
        selected_features += [vectorizer.get_feature_names()[i] for i in selected_indices]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

        # Train a decision tree classifier
        dt = DecisionTreeClassifier(random_state=1)
        param_grid = {'max_depth': [5, 10, 15, 20], 'criterion': ['gini', 'entropy']}
        clf = GridSearchCV(dt, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train[:, selected_indices], y_train)

        # Evaluate the model on the testing set
        y_pred = clf.predict(X_test[:, selected_indices])
        print_classification_metrics(y_test, y_pred, target_names=labels)
        print('Selected features:', ', '.join([vectorizer.get_feature_names()[i] for i in selected_indices]))
        print('')

    # Perform multi-class classification using selected features
    top_k *= 4  # Increase the number of top features to include all selected features from binary classification tasks
    selected_features = list(set(selected_features))[:top_k]
    print('Performing multi-class classification using top {} features:'.format(top_k))
    X = df['Message'].values
    y = df['Label'].values

    # Convert text data to TF-IDF features
    vectorizer =  TfidfVectorizer(analyzer='word',vocabulary=selected_features, stop_words='english')
    X = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Train a decision tree classifier
    dt = DecisionTreeClassifier(random_state=1)
    param_grid = {'max_depth': [5, 10, 15, 20], 'criterion': ['gini', 'entropy']}
    clf = GridSearchCV(dt, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = clf.predict(X_test)

    # Get the unique label names in the order of the classes
    label_list = clf.classes_

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=label_list)

    # Create a heatmap of the confusion matrix with label names
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Multi-class Classification')
    plt.show()

    # Print classification report
    print('Classification Report:')
    print_classification_metrics(y_test, y_pred, target_names=label_list)
    print('Top 200 features:')
    for i, feature in enumerate(feature_names[:200]):
        print(i+1, feature)


from sklearn.ensemble import RandomForestClassifier

def perform_binary_random_forest_classification(df, top_k=50):
    # Define binary classification tasks
    binary_tasks = {
        'Info-giving vs. Non-Info-giving': ['Info Giving', 'Non-Info-giving'],
        'Info-seeking vs. Non-info-seeking': ['Info Seeking', 'Non-info-seeking'],
        'Emotion vs. non-emotion': ['Emotion', 'Non-emotion'],
        'Partnership vs. non-partnership': ['Partnership', 'Non-partnership']
    }

    # Perform binary classification tasks
    selected_features = []
    for task_name, labels in binary_tasks.items():
        print('Performing binary classification for task:', task_name)
        X = df['Message'].values
        y = df['Label'].apply(lambda x: labels[0] if x == labels[0] else labels[1]).values

        # Convert text data to TF-IDF features
        vectorizer =   TfidfVectorizer(analyzer='word', stop_words='english')
        X = vectorizer.fit_transform(X)

        # Perform feature selection using chi-square test
        selected_indices, p_values = chi2_feature_selection(X, y, top_k)
        selected_features += [vectorizer.get_feature_names()[i] for i in selected_indices]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

        # Train a random forest classifier
        rf = RandomForestClassifier(random_state=1)
        param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [10, 20, 30], 'criterion': ['gini', 'entropy']}
        clf = GridSearchCV(rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train[:, selected_indices], y_train)

        # Evaluate the model on the testing set
        y_pred = clf.predict(X_test[:, selected_indices])
        print_classification_metrics(y_test, y_pred, target_names=labels)
        print('Selected features:', ', '.join([vectorizer.get_feature_names()[i] for i in selected_indices]))
        print('')

    # Perform multi-class classification using selected features
    top_k *= 4  # Increase the number of top features to include all selected features from binary classification tasks
    selected_features = list(set(selected_features))[:top_k]
    print('Performing multi-class classification using top {} features:'.format(top_k))
    X = df['Message'].values
    y = df['Label'].values

    # Convert text data to TF-IDF features
    vectorizer =  TfidfVectorizer(analyzer='word',vocabulary=selected_features, stop_words='english')
    X = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Train a random forest classifier
    rf = RandomForestClassifier(random_state=1)
    param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [10, 20, 30], 'criterion': ['gini', 'entropy']}
    clf = GridSearchCV(rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = clf.predict(X_test)

    # Get the unique label names in the order of the classes
    label_list = clf.classes_

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=label_list)

    # Create a heatmap of the confusion matrix with label names
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Multi-class Classification')
    plt.show()

    # Print classification report
    print('Classification Report:')
    print_classification_metrics(y_test, y_pred, target_names=label_list)
    print('Top 200 features:')
    for i, feature in enumerate(feature_names[:200]):
        print(i+1, feature)

from sklearn.ensemble import GradientBoostingClassifier

def perform_binary_gradient_boosting_classification(df, top_k=50):
    # Define binary classification tasks
    binary_tasks = {
        'Info-giving vs. Non-Info-giving': ['Info Giving', 'Non-Info-giving'],
        'Info-seeking vs. Non-info-seeking': ['Info Seeking', 'Non-info-seeking'],
        'Emotion vs. non-emotion': ['Emotion', 'Non-emotion'],
        'Partnership vs. non-partnership': ['Partnership', 'Non-partnership']
    }

    # Perform binary classification tasks
    selected_features = []
    for task_name, labels in binary_tasks.items():
        print('Performing binary classification for task:', task_name)
        X = df['Message'].values
        y = df['Label'].apply(lambda x: labels[0] if x == labels[0] else labels[1]).values

        # Convert text data to TF-IDF features
        vectorizer =  TfidfVectorizer(analyzer='word', stop_words='english')
        X = vectorizer.fit_transform(X)

        # Perform feature selection using chi-square test
        selected_indices, p_values = chi2_feature_selection(X, y, top_k)
        selected_features += [vectorizer.get_feature_names()[i] for i in selected_indices]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

        # Train a gradient boosting classifier
        gbm = GradientBoostingClassifier()
        param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01], 'max_depth': [3, 4, 5]}
        clf = GridSearchCV(gbm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train[:, selected_indices], y_train)

        # Evaluate the model on the testing set
        y_pred = clf.predict(X_test[:, selected_indices])
        print_classification_metrics(y_test, y_pred, target_names=labels)
        print('Selected features:', ', '.join([vectorizer.get_feature_names()[i] for i in selected_indices]))
        print('')

    # Perform multi-class classification using selected features
    top_k *= 4  # Increase the number of top features to include all selected features from binary classification tasks
    selected_features = list(set(selected_features))[:top_k]
    print('Performing multi-class classification using top {} features:'.format(top_k))
    X = df['Message'].values
    y = df['Label'].values

    # Convert text data to TF-IDF features
    vectorizer =  TfidfVectorizer(analyzer='word', vocabulary=selected_features, stop_words='english')
    X = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Train a gradient boosting classifier
    gbm = GradientBoostingClassifier()
    param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01], 'max_depth': [3, 4, 5]}
    clf = GridSearchCV(gbm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)
    # Evaluate the model on the testing set
    y_pred = clf.predict(X_test)

    # Get the unique label names in the order of the classes
    label_list = clf.classes_

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=label_list)

    # Create a heatmap of the confusion matrix with label names
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Multi-class Classification')
    plt.show()

    # Print classification report
    print('Classification Report:')
    print_classification_metrics(y_test, y_pred, target_names=label_list)
    print('Top 200 features:')
    for i, feature in enumerate(feature_names[:200]):
        print(i+1, feature)

def perform_logistic_regression(df):
    X = df['Message'].values
    y = df['Label'].values

    # Convert text data to TF-IDF features
    vectorizer =  TfidfVectorizer(analyzer='word', stop_words='english')
    X = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Train a logistic regression model
    lr = LogisticRegression(max_iter=1000)
    param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
    clf = GridSearchCV(lr, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = clf.predict(X_test)

    # Get the unique label names in the order of the classes
    label_list = clf.classes_

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=label_list)

    # Create a heatmap of the confusion matrix with label names
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Multi-class Classification')
    plt.show()

    # Print classification report
    print('Classification Report:')
    print_classification_metrics(y_test, y_pred, target_names=label_list)
    print('Top 200 features:')
    for i, feature in enumerate(feature_names[:200]):
        print(i+1, feature)

from sklearn.tree import DecisionTreeClassifier

def perform_decision_tree_classification(df):
    X = df['Message'].values
    y = df['Label'].values

    # Convert text data to TF-IDF features
    vectorizer =  TfidfVectorizer(analyzer='word', stop_words='english')
    X = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Train a decision tree classifier
    dt = DecisionTreeClassifier(random_state=1)
    param_grid = {'max_depth': [5, 10, 15, 20], 'criterion': ['gini', 'entropy']}
    clf = GridSearchCV(dt, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = clf.predict(X_test)

    # Get the unique label names in the order of the classes
    label_list = clf.classes_

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=label_list)

    # Create a heatmap of the confusion matrix with label names
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Multi-class Classification')
    plt.show()

    # Print classification report
    print('Classification Report:')
    print_classification_metrics(y_test, y_pred, target_names=label_list)
    print('Top 200 features:')
    for i, feature in enumerate(feature_names[:200]):
        print(i+1, feature)


from sklearn.ensemble import RandomForestClassifier

def perform_random_forest_classification(df):
    X = df['Message'].values
    y = df['Label'].values

    # Convert text data to TF-IDF features
    vectorizer =  TfidfVectorizer(analyzer='word', stop_words='english')
    X = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Train a random forest classifier
    rf = RandomForestClassifier(random_state=1)
    param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [10, 20, 30], 'criterion': ['gini', 'entropy']}
    clf = GridSearchCV(rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = clf.predict(X_test)

    # Get the unique label names in the order of the classes
    label_list = clf.classes_

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=label_list)

    # Create a heatmap of the confusion matrix with label names
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Multi-class Classification')
    plt.show()

    # Print classification report
    print('Classification Report:')
    print_classification_metrics(y_test, y_pred, target_names=label_list)
    print('Top 200 features:')
    for i, feature in enumerate(feature_names[:200]):
        print(i+1, feature)

from sklearn.ensemble import GradientBoostingClassifier

def perform_gradient_boosting_classification(df):
    X = df['Message'].values
    y = df['Label'].values

    # Convert text data to TF-IDF features
    vectorizer =  TfidfVectorizer(analyzer='word',stop_words='english')
    X = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Train a gradient boosting classifier
    gbm = GradientBoostingClassifier()
    param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01], 'max_depth': [3, 4, 5]}
    clf = GridSearchCV(gbm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)
    # Evaluate the model on the testing set
    y_pred = clf.predict(X_test)

    # Get the unique label names in the order of the classes
    label_list = clf.classes_

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=label_list)

    # Create a heatmap of the confusion matrix with label names
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Multi-class Classification')
    plt.show()

    # Print classification report
    print('Classification Report:')
    print_classification_metrics(y_test, y_pred, target_names=label_list)
    print('Top 200 features:')
    for i, feature in enumerate(feature_names[:200]):
        print(i+1, feature)