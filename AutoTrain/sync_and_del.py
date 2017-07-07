import subprocess
import sys
SERVER_SSH = 'greenhouse@k-yantra.org'
SERVER_DIR = '/var/www/Ghfarm/django/Autotrain/'
LOCAL_DIR = '/home/weighingscale/AutoTrain/Images/'

sync_str = 'rsync -auvz ' + SERVER_SSH + ':' + SERVER_DIR + ' ' + LOCAL_DIR

try:
    process = subprocess.check_call(sync_str.split(), stdout=subprocess.PIPE)
except Exception as e:
    print(e)
    sys.exit()
output, error = process.communicate()
print(output, error)

rm_str = 'ssh ' + SERVER_SSH ' rm -r ' + SERVER_DIR

try:
    process = subprocess.check_call(sync_str.split(), stdout=subprocess.PIPE)
except Exception as e:
    print(e)
output, error = process.communicate()
print(output, error)