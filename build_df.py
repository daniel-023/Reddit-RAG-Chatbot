import pandas as pd
import praw
import os
from datetime import datetime, timezone, timedelta


def initialize_reddit_scraper():
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )
    return reddit


def scrape_posts(reddit, subreddit_name, start_date, end_date, limit = 1000):
    subreddit = reddit.subreddit(subreddit_name)
    data = []

    for post in subreddit.top(time_filter="all", limit=limit):
        post_datetime = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
        if start_date <= post_datetime <= end_date:
            data.append({
                'Type': 'Post',
                'Post_id': post.id,
                'Title': post.title,
                'Author': post.author.name if post.author else 'Unknown',
                'Timestamp': post_datetime,
                'Text': post.selftext,
                'Score': post.score,
                'Total_comments': post.num_comments,
                'Post_URL': post.url
            })

            if post.num_comments > 0:
                post.comments.replace_more(limit=None)
                for comment in post.comments.list():
                    comment_timestamp = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)
                    data.append({
                        'Type': 'Comment',
                        'Post_id': post.id,
                        'Title': post.title,
                        'Author': comment.author.name if comment.author else 'Unknown',
                        'Timestamp': comment_timestamp,
                        'Text': comment.body,
                        'Score': comment.score,
                        'Total_comments': 0, #Comments don't have this attribute
                        'Post_URL': None  #Comments don't have this attribute
                    })
    return pd.DataFrame(data)


def update_reddit_data():
    try:
        df = pd.read_csv('data/reddit_data.csv')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
    except FileNotFoundError:
        df = pd.DataFrame()
        
    reddit = initialize_reddit_scraper()

    # Scrape posts from the last month
    end_date = datetime.now(timezone.utc)
    start_date = end_date-timedelta(days=30)

    new_data = scrape_posts(reddit, 'NTU', start_date, end_date)

    if not new_data.empty:
        df = pd.concat([df, new_data], ignore_index=True)
    
    one_year_ago = end_date - timedelta(days=365)
    df = df[df['Timestamp'] >= one_year_ago]

    df.to_csv('data/reddit_data.csv', index=False)


if __name__ == "__main__":
    update_reddit_data()