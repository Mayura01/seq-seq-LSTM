import json
import pymongo
import re
import nltk
from nltk.corpus import words


print("Connecting...")
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["reddit_dataset"]
print("Connected!")

nltk.download('words')
english_words = set(words.words())

def clean_and_insert(data):
    cleaned_text = clean_text(data['body'])
    if not cleaned_text or "http" in cleaned_text or '[deleted]' in cleaned_text or '\n' in cleaned_text or not contains_english_words(cleaned_text):
        return False
    cleaned_data = {
        'body': cleaned_text,
    }
    return cleaned_data

def contains_english_words(text):
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token.lower() in english_words:
            return True
    return False

def store_comments(comments, collection_name):
    collection = db[collection_name]
    collection.insert_many(comments)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

def main():
    chunk_size = 10000
    current_chunk = 1
    comments = []

    with open("RC_2015-01", "r") as file:
        for line in file:
            try:
                conversation = json.loads(line.strip())
                comment = clean_and_insert(conversation)
            except json.JSONDecodeError as e:
                print("Error parsing line:", e)
                continue

            if comment:
                comments.append(comment)

            if len(comments) >= chunk_size:
                print("current chunk: ", current_chunk)
                store_comments(comments, f"comments_chunk_{current_chunk}")
                print(f"Chunk {current_chunk} stored.")
                current_chunk += 1
                comments = []

    if comments:
        store_comments(comments, f"comments_chunk_{current_chunk}")
        print(f"Chunk {current_chunk} stored.")

    print("data inserted")

if __name__ == "__main__":
    main()
