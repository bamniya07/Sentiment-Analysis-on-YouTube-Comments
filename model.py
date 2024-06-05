from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from string import punctuation
import nltk
import re
import pickle
import pandas as pd


data = pd.read_csv("Dataset.csv")
data1 = data.drop(['Comment_ID', 'Reply_Count', 'Like_Count', 'Date', 'VidId', 'user_ID'], axis=1)
# print(data1)

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def TextBlob(text):

    sentiments = SentimentIntensityAnalyzer()
    text["Positive"] = [sentiments.polarity_scores(str(i))["pos"] for i in text["Comments"]]
    text["Negative"] = [sentiments.polarity_scores(str(i))["neg"] for i in text["Comments"]]
    text["Neutral"] = [sentiments.polarity_scores(str(i))["neu"] for i in text["Comments"]]
    text['Compound'] = [sentiments.polarity_scores(str(i))["compound"] for i in text["Comments"]]

    # Assign sentiment labels
    sentiment = []
    for i in text["Compound"]:
        if i >= 0.05:
            sentiment.append(
                'Positive')
        elif i <= -0.05:
            sentiment.append('Negative')
        else:
            sentiment.append('Neutral')
    text["Sentiment"] = sentiment

    # Print the updated DataFrame
    #print(text.head(25))

    data2=text.drop(['Positive','Negative','Neutral','Compound'],axis=1)
    return data2

data2=TextBlob(data1)

nltk.download('stopwords')
stop_words = stopwords.words('english')
lzr = WordNetLemmatizer()

def text_processing(text):
    # convert text into lowercase
    text = text.lower()

    # remove new line characters in text
    text = re.sub(r'\n',' ', text)

    # remove punctuations from text
    text = re.sub('[%s]' % re.escape(punctuation), "", text)

    # remove references and hashtags from text
    text = re.sub("^a-zA-Z0-9$,.", "", text)

    # remove multiple spaces from text
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # remove special characters from text
    text = re.sub(r'\W', ' ', text)

    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])

    # lemmatizer using WordNetLemmatizer from nltk package
    text=' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])

    return text

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')

def processdata(data2):
    # Assuming 'data2' contains the 'Comments' column

    data_copy = data2.copy()

    # Convert any missing values (NaN) to an empty string
    data_copy['Comments'] = data_copy['Comments'].fillna('')

    # Apply the 'text_processing' function to the 'Comments' column
    data_copy['Comments'] = data_copy['Comments'].apply(lambda text: text_processing(str(text)))

     # Assign sentiment labels based on the individual polarity scores
    # data_copy['Sentiment'] = data_copy.apply(
    #     # lambda row: 'Positive' if row['Positive'] > row['Negative'] and row['Positive'] > row['Neutral'] else
    #     #             'Negative' if row['Negative'] > row['Positive'] and row['Negative'] > row['Neutral'] else
    #     #             'Neutral',
    #     # axis=1
    # )
    le = LabelEncoder()
    data_copy['Sentiment'] = le.fit_transform(data_copy['Sentiment'])

    processed_data = {
        'Sentence':data_copy.Comments,
        'Sentiment':data_copy['Sentiment']
    }

    processed_data = pd.DataFrame(processed_data)
    #processed_data.head()

    processed_data['Sentiment'].value_counts()

   
    df_neutral = processed_data[(processed_data['Sentiment']==1)]
    df_negative = processed_data[(processed_data['Sentiment']==0)]
    df_positive = processed_data[(processed_data['Sentiment']==2)]

    # upsample minority classes
    df_negative_upsampled = resample(df_negative,
                                     replace=True,
                                     n_samples= 2500,
                                     random_state=42)

    df_neutral_upsampled = resample(df_neutral,
                                     replace=True,
                                     n_samples= 2500,
                                     random_state=42)
    # downsampling the majority positive sentiment

    
    df_positive_downsampled = resample(df_positive,
                                     replace=True,
                                     n_samples= 2500,
                                     random_state=42)


    # Concatenate the upsampled dataframes with the neutral dataframe
    final_data = pd.concat([df_negative_upsampled,df_neutral_upsampled,df_positive_downsampled])

    return final_data

final_data = processdata(data2)

def finalsentence(final_data):
    corpus = []
    for sentence in final_data['Sentence']:
        corpus.append(sentence)
    #corpus[0:5]
    return corpus

corpus = finalsentence(final_data)
  
cv = CountVectorizer(max_features=1500)
  

X = cv.fit_transform(corpus).toarray()
y = final_data.iloc[:, -1].values

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
RF_score = accuracy_score(y_test, y_pred)
print(RF_score)
print("SUCCESS")


pickle.dump(cv, open('cv1.pkl', 'wb'))
 
# dump a model in a pickle file
pickle.dump(classifier, open('model.pkl', 'wb'))   
