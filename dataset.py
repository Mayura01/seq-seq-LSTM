import json
import pymongo

print("Connecting...")
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["reddit_dataset"]
print("Connected!")

def clean_data(comment):
    if not comment["body"] or "http" in comment["body"] or '[deleted]' in comment["body"]:
        return False
    return True

def clean_and_insert(data):
    cleaned_data = {
        'author': data['author'],
        'body': data['body'],
        'subreddit': data['subreddit'],
        'created_utc': data['created_utc']
    }
    return cleaned_data

def store_comments(comments, collection_name):
    collection = db[collection_name]
    collection.insert_many(comments)

def main():
    chunk_size = 25000
    current_chunk = 1
    comments = []

    with open("RC_2015-01", "r") as file:
        for line in file:
            try:
                conversation = json.loads(line.strip())
                comment = clean_and_insert(conversation)
            except Exception as e:
                print("Error parsing line:", e)

            if clean_data(comment):
                comments.append(comment)

            if len(comments) >= chunk_size:
                print("current chunk: ",current_chunk)
                store_comments(comments, f"comments_chunk_{current_chunk}")
                current_chunk += 1
                comments = []

    # remaining comments
    if comments:
        store_comments(comments, f"comments_chunk_{current_chunk}")

    print("data inserted")

if __name__ == "__main__":
    main()
