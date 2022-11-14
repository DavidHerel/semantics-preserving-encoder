import io
import pathlib
import urllib.error
import urllib.request
import zipfile

default_classifiers_url = "https://data.ciirc.cvut.cz/public/projects/2022PreservingSemanticsEncoder/default_classifiers.zip"


def download_default_classifiers(
    extract_to: pathlib.Path = pathlib.Path("./spe_classifiers"),
):
    print("Downloading default classifiers.")
    try:
        with urllib.request.urlopen(
            default_classifiers_url
        ) as http_response, zipfile.ZipFile(
            io.BytesIO(http_response.read())
        ) as zip_file:
            zip_file.extractall(path=extract_to)
        print("Done.")

    except urllib.error.URLError as e:
        print(f"Error opening default classifiers URL: {e}")

    except zipfile.BadZipFile as e:
        print(f"Error downloading default classifiers zip archive: {e}")
