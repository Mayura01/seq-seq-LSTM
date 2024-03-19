import pymongo

print("Connecting...")
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["Cornell_Movie_Dialog_Corpus"]
print("Connected!")

def split_dataset_into_qa(dataset_file):
    questions = []
    answers = []

    with open(dataset_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                q, a = line.split('\t')
                questions.append(q.strip())
                answers.append(a.strip())

    return questions, answers

# Example usage:
dataset_file = "dialogs.txt"
questions, answers = split_dataset_into_qa(dataset_file)

print(len(questions))
print(len(answers))

# Print the first few questions and answers to verify
for i in range(5):  # Adjust the range as needed
    print("Question:", questions[i])
    print("Answer:", answers[i])
    print()


# collection = db["Dialog_Corpus"]

# for i in range(len(questions)):
#     data = {
#         "question": questions[i],
#         "answer": answers[i]
#     }
#     collection.insert_one(data)

# print("Data insertion completed.")