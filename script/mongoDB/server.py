import subprocess
import os

if __name__ == '__main__':
    child = None
    try:
        db_path = os.path.join('.', 'data', 'db')
        port = str(1234)

        args = ['mongod.exe', '--dbpath', db_path, '--port', port]
        child = subprocess.Popen(args, shell=False)
        child.wait()
    except BaseException as e:
        child.kill()
        pass
