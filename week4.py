# Installing and set up Kaggle API
!pip install kaggle

# Uploading kaggle (API key file)
from google.colab import files
print("Please upload your kaggle.json file now:")
files.upload()

# Moving it to the right folder
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Downloading the dataset (Students Performance)
!kaggle datasets download -d rkiattisak/student-performance-in-english

# Step 5: Unzip it
!unzip student-performance-in-english.zip

# Loading the dataset using pandas
import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")
print(df.head())  # Just to preview the data

# Converting DataFrame to a list of dictionaries
students_list = df.to_dict(orient="records")

# Writting a simple function to sort by a specific key
def sort_students(data, key):
    """
    Sorts a list of dictionaries by the given key.
    Example: sort_students(students_list, 'english score')
    """
    return sorted(data, key=lambda x: x[key])

# Testing the function
sorted_students = sort_students(students_list, 'english score')

# Displaying first 5 results
for student in sorted_students[:5]:
    print(student)