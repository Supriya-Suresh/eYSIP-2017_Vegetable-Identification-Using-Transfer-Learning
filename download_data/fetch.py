from urllib.request import urlretrieve
import os
import socket

socket.setdefaulttimeout(20)
input_file_name = os.getcwd() + os.sep + os.getcwd().split(os.sep)[-1] + '.txt'
log_file_name = 'log.txt'
dummy_file_name = 'dummy.txt'


log_file = open(log_file_name,'w')
with open(input_file_name,'r') as input_file:
    for line in input_file:
        if not line.strip():
            break
        log_file.write(line)
        lineFailed = False
        line = line.replace('\n', '')
        URL = line
        IMAGE = URL.rsplit('/',1)[1]
        try:
            urlretrieve(URL, IMAGE)
        except KeyboardInterrupt:
            log_file.close()
            with open(log_file_name,'r') as log_file:
                already_downloaded_lines = log_file.readlines()
                with open(dummy_file_name,'w') as dummy_file:
                    for line in input_file:
                        if line not in already_downloaded_lines:
                            dummy_file.write(line)
            os.rename(input_file_name,'d.txt')
            os.rename(dummy_file_name,input_file_name)
            os.remove(log_file_name)
            os.remove('d.txt')
            raise
        except Exception as e:
            print("Failed on:", URL, "with","{}".format(e))
            lineFailed = True
        if not lineFailed:
            print("Successfully Downloaded:",URL)
log_file.close()