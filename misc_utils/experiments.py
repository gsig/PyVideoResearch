import os
import subprocess


def get_script_dir_commit_hash():
    current_working_dir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    try:
        commit = subprocess.check_output(["git", "describe", "--always"]).strip()
        return commit.decode('UTF-8')
    except Exception as e:
        print(e)
        return ''
    finally:
        os.chdir(current_working_dir)


def experiment_checksums():
    try:
        result = subprocess.check_output(["find", "-type", "f", "!", "-name", "*.pyc", "-exec", "md5sum", '{}', ';'])
        return result.decode('UTF-8')
    except Exception as e:
        print(e)
        return ''


def experiment_folder():
    current_working_dir = os.getcwd()
    return current_working_dir
