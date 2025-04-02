plays = {
    "Anthony and Cleopatra": "Anthony is there, Brutus is Caeser is with Cleopatra mercy worser.",
    "Julius Caeser": "Anthony is there, Brutus is Caeser is but Calpurnia is.",
    "The Tempest": "mercy worser",
    "Hamlet": "Caeser and Brutus are present with mercy and worser",
    "Othello": "Caeser is present with mercy",
    "Macbeth": "Anthony is there; Caeser, mercy."
}
words = ["Anthony", "Brutus", "Caeser", "Calpurnia", "Cleopatra", "mercy", "worser", ]
vector_matrix = [[0 for _ in range(len(plays))] for _ in range(len(words))]
text_list = list(plays.values())
# Creating the vector matrix
for i in range(len(words)):
    for j in range(len(text_list)):
        if words[i] in text_list[j]:
            vector_matrix[i][j] = 1
        else:
            vector_matrix[i][j] = 0
#Printing the vector matrix with headings
print(f"{'Words/Plays':<15}", end="\t")  # Print the header for the plays
for play in plays.keys():
    print(f"{play:<18}", end="\t")  # Adjust width for play names
print()  # New line after the header
for i in range(len(words)):
    print(f"{words[i]:<12}", end="\t")  # Print the word with adjusted width
    for j in range(len(text_list)):
        print(f"{vector_matrix[i][j]:<18}", end="\t")  # Print the vector values with adjusted width
    print()  # New line after each word row
vector_dict = {}
# Generating binary presence as integers
for vector in range(len(vector_matrix)):
    mystring = ""
    for digit in vector_matrix[vector]:
        mystring += str(digit)
    vector_dict[words[vector]] = int(mystring, 2)
print("\n",vector_dict,"\n")
# Taking condition as input
condition = input("Enter your condition: ")
condition1 = condition
# Replace words with their corresponding binary integer representation
for i in words:
    if i in condition1:
        condition1 = condition1.replace(i, str(vector_dict[i]))
print(condition1)
# Replace logical operators for evaluation
condition1 = condition1.replace("not", "and~").replace("or", "|").replace("and", "&")
# Evaluate the condition with error handling
try:
    stri = bin(eval(condition1)).replace("0b", "")  # Evaluate and convert to binary string
    print(stri)
    print("The plays which satisfy the condition", condition, "are:")
    for char in range(len(stri)):
        if stri[char] == "1":
            print(list(plays.keys())[char])
except NameError as e:
    print(f"Error: {e}. Please ensure your condition only uses defined words: {', '.join(words)}.")
except Exception as e:
    print(f"An error occurred: {e}")
