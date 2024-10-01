import pandas as pd
import praw
import os
from datetime import datetime, timezone, timedelta


def initialize_reddit_scraper():
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')

    if not client_id:
        raise ValueError("REDDIT_CLIENT_ID environment variable not set")
    if not client_secret:
        raise ValueError("REDDIT_CLIENT_SECRET environment variable not set")
    if not user_agent:
        raise ValueError("REDDIT_USER_AGENT environment variable not set")

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    return reddit


def scrape_posts(reddit, subreddit_name, start_date, end_date, limit = 1000):
    subreddit = reddit.subreddit(subreddit_name)
    data = []

    for post in subreddit.top(time_filter="month", limit=limit):
        post_datetime = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
        if start_date <= post_datetime <= end_date:
            data.append({
                'Type': 'Post',
                'Post_id': post.id,
                'Comment_id': None,  
                'Title': post.title,
                'Author': getattr(post.author, 'name', 'Unknown') if post.author else 'Unknown',
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
                    if start_date <= post_datetime <= end_date:
                        data.append({
                            'Type': 'Comment',
                            'Post_id': post.id,
                            'Comment_id': comment.id,
                            'Title': post.title,
                            'Author': getattr(comment.author, 'name', 'Unknown') if comment.author else 'Unknown', 
                            'Timestamp': comment_timestamp,
                            'Text': comment.body,
                            'Score': comment.score,
                            'Total_comments': 0, #Comments don't have this attribute
                            'Post_URL': None  #Comments don't have this attribute
                        })
    return pd.DataFrame(data)


def update_reddit_data(subreddit_name='NTU'):
    os.makedirs('data', exist_ok=True)

    try:
        df = pd.read_csv('data/reddit_data.csv')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    except FileNotFoundError:
        df = pd.DataFrame()
        
    reddit = initialize_reddit_scraper()

    # Scrape posts from the last month
    end_date = datetime.now(timezone.utc)
    start_date = end_date-timedelta(days=30)

    new_data = scrape_posts(reddit, subreddit_name, start_date, end_date)

    if not new_data.empty:
        df = pd.concat([df, new_data], ignore_index=True)
        df.drop_duplicates(subset=['Type', 'Post_id', 'Comment_id'], inplace=True)

    one_year_ago = end_date - timedelta(days=365)
    df = df[df['Timestamp'] >= one_year_ago]

    df.to_csv('data/reddit_data.csv', index=False)


if __name__ == "__main__":
    update_reddit_data(subreddit_name='NTU')