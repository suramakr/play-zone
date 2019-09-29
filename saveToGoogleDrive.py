# Import PyDrive and associated libraries.
# This only needs to be done once in a notebook.
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once in a notebook.
auth.authenticate_user()
# https://pythonhosted.org/PyDrive/quickstart.html
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Create & upload a text file.
uploaded = drive.CreateFile({'title': 'Sample file.txt'})
uploaded.SetContentString('Sample upload file content')
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))


# listing all files
# List .txt files in the root.
#
# Search query reference:
# https://developers.google.com/drive/v2/web/search-parameters
listed = drive.ListFile(
    {'q': "title contains '.txt' and 'root' in parents"}).GetList()
for file in listed:
    print('title {}, id {}'.format(file['title'], file['id']))


# Download a file based on its file ID.
#
# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
# https://drive.google.com/file/d/1ILCp8GmHtBS1qA8cyyqE5L7okJGJhvpt/view?usp=sharing
file_id = '1ILCp8GmHtBS1qA8cyyqE5L7okJGJhvpt'
downloaded = drive.CreateFile({'id': file_id})
print('Downloaded content "{}"'.format(downloaded.GetContentString()))
# to save the content to a file
downloaded.GetContentFile("mobile.txt")


# if you want to now read a file stored in drive, and that has been copied
# over locally, you can then do this.
df = pd.read_csv('mobile_cleaned_local.csv')
df.head()
