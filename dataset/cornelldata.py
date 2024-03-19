import re
import pymongo

print("Connecting...")
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["Cornell_Movie_Dialog_Corpus"]
print("Connected!")

# Load the data
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# Create a dictionary to map each line's id with its text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Create a list of all conversations' lines' ids
convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))

# Sort the sentences into questions (inputs) and answers (targets)
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])

def split_dataset_into_qa(dataset_file):
    with open(dataset_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                q, a = line.split('\t')
                questions.append(q.strip())
                answers.append(a.strip())

dataset_file = "dialogs.txt"

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
    text = " ".join(text.split())
    return text

# Clean the data
clean_questions = [clean_text(question) for question in questions]
clean_answers = [clean_text(answer) for answer in answers]

# Function to store comments in chunks
def store_comments(comments, collection_name):
    collection = db[collection_name]
    collection.insert_many(comments)

# Chunk size
chunk_size = 10000
num_chunks = (len(clean_questions) + chunk_size - 1) // chunk_size

for chunk_num in range(num_chunks):
    start_index = chunk_num * chunk_size
    end_index = min((chunk_num + 1) * chunk_size, len(clean_questions))
    
    comments = []
    for i in range(start_index, end_index):
        data = {
            "question": clean_questions[i],
            "answer": clean_answers[i]
        }
        comments.append(data)

    collection_name = f"Dialog_Corpus_{chunk_num + 1}"
    store_comments(comments, collection_name)
    print(f"Chunk {chunk_num + 1} stored.")

print("Data insertion completed.")
