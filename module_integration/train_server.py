import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import json
import os

hostName = "192.168.0.66"
hostPort = 6969
SECRET_KEY = 'El-Psy-Kongroo'
TRAIN_DIRECTORY = '/home/teknas/Desktop/Train/'
# Expected payload keys
EXPECTED_KEYS = ['secret','image','image_name','label']

class TrainServer(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
    
    def _deny_access(self,code):
        self.send_response(code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(bytes("<html><body><h1>Get Request Received!</h1></body></html>",'utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        formatted_data = json.loads(post_data.decode('utf-8'))

        if all(key in formatted_data for key in EXPECTED_KEYS):
            if formatted_data['secret'] == SECRET_KEY:
                
                # Folder to store image
                class_folder = os.path.join(TRAIN_DIRECTORY,formatted_data['label'])
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
                
                # Write image to folder
                image_path = os.path.join(class_folder,formatted_data['image_name'])
                with open(image_path,'wb') as img_file:
                    img_file.write(bytes(formatted_data['image'],'latin-1'))
                
                print("Successfully written image at location: " + image_path)
                
                # Send success status
                self._set_headers()
            else:
                # Wrong secret key
                self._deny_access(401)
        else:
            # Some Data Missing
            self._deny_access(400)


if __name__ == '__main__':

    if not os.path.exists(TRAIN_DIRECTORY):
        os.makedirs(TRAIN_DIRECTORY)

    trainServer = HTTPServer((hostName, hostPort), TrainServer)
    print(time.asctime(), "Server Starting on - %s:%s" % (hostName, hostPort))

    try:
        trainServer.serve_forever()
    except KeyboardInterrupt:
        trainServer.server_close()
        print(time.asctime(), "Server Stopped - %s:%s" % (hostName, hostPort))