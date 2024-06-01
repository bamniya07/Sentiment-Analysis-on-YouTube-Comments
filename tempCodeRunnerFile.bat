<!DOCTYPE html>
<html>
<head>
    <title>YouTube Comment Sentiment Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            text-align: center;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .comment {
            border: 1px solid #ccc;
            padding: 10px;
            margin-bottom: 20px;
        }
        .comment .sentiment {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .comment .text {
            margin-bottom: 10px;
        }
        .result-summary {
            margin-bottom: 20px;
        }
        .result-summary p {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <h1>YouTube Comment Sentiment Analysis</h1>
    <div class="container">
        <div class="result-summary">
            <h2>Analysis Summary</h2>
            <p>Total comments analyzed: {{ total_comments }}</p>
            <p>Positive comments: {{ positive_comments }} ({{ positive_percentage }}%)</p>
            <p>Negative comments: {{ negative_comments }} ({{ negative_percentage }}%)</p>
            <p>Neutral comments: {{ neutral_comments }} ({{ neutral_percentage }}%)</p>
        </div>

        <h2>Detailed Results</h2>
        {% for comment, sentiment in zip(comments, sentiments) %}
        <div class="comment">
            <div class="sentiment">Sentiment: {{ sentiment }}</div>
            <div class="text">{{ comment }}</div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
