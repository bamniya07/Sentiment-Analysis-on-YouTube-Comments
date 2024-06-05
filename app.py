from flask import Flask, request, jsonify, render_template
from model import TextBlob, text_processing
import pickle
import nltk
from googleapiclient.discovery import build
#from oauth2client.tools import argparser


model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv1.pkl', 'rb'))

app = Flask(__name__,template_folder='template')

API_KEY = "AIzaSyB6TukTDBX3BQcYJtFkTNQBztWCV4eMUvE"
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

def get_comments(video_id):
    comments = []
    next_page_token = None
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",

            pageToken=next_page_token
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break  # No more pages

    return comments

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Get the YouTube video link from the form
        video_link = request.form['video_link']
        video_id = video_link.split('be/')[1].split('?')[0]

         # Scrape comments from the YouTube video
        comments = get_comments(video_id)
        processed_comments = [text_processing(c) for c in comments]
        X = cv.transform(processed_comments).toarray()
        predictions = model.predict(X)
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        sentiments = [sentiment_labels[pred] for pred in predictions]
        
        # Calculate the sentiment distribution
        total_comments = len(comments)
        positive_comments = sentiments.count('Positive')
        negative_comments = sentiments.count('Negative')
        neutral_comments = sentiments.count('Neutral')
        positive_percentage = (positive_comments / total_comments) * 100
        negative_percentage = (negative_comments / total_comments) * 100
        neutral_percentage = (neutral_comments / total_comments) * 100
        
        if len(comments) > len(sentiments):
            comments = comments[:len(sentiments)]
        elif len(sentiments) > len(comments):
            sentiments = sentiments[:len(comments)]

        return render_template('result.html', total_comments=total_comments,
                              positive_comments=positive_comments,
                              negative_comments=negative_comments,
                              neutral_comments=neutral_comments,
                              positive_percentage=positive_percentage,
                              negative_percentage=negative_percentage,
                              neutral_percentage=neutral_percentage,
                              comments = comments,
                              sentiments = sentiments)
    else:
        # Preprocess the input comment
        processed_comment = text_processing(comments)

        # Convert the preprocessed comment to a feature vector
        X = cv.transform([processed_comment]).toarray()

        # Use the pre-trained model to predict the sentiment
        prediction = model.predict(X)[0]

        # Map the predicted label back to the original sentiment labels
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        sentiment = sentiment_labels[prediction]

        return render_template('result.html')
    

if __name__ == '__main__':
    app.run(debug=True)

