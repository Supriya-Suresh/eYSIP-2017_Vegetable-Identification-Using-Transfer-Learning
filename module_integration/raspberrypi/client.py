import requests  
import json
import datetime
import os
import constants

if os.path.getsize(constants.data_offline) > 0:
	crop_details = []
	pending = []
	#time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	with open(constants.data_offline, 'r') as f:
		crop_details = list(f.readlines())
	last = crop_details[len(crop_details) - 1]
	i=1
	print(crop_details)
	for details in crop_details:
		values = json.loads(details)
		values['user_id']=1
		values['password']='random'
		values['upload']='offline'
		with open(constants.imagepath + values['imagename'], 'rb') as img:
			image = img.read()
			image = str(image,"latin-1")
			print("Image Size: ",len(image)/1024," KB")
		values['image']=image
		i=i+1
		try:
			r = requests.post(constants.URL, data=json.dumps(values))
			print(r.text)
			if(r.text != "Done"):
				pending.append(details)
		except:
			pending.append(details)
			print("Error occured")

	with open(constants.data_offline, 'w') as detail_file:
		for details in pending:
			detail_file.write(details)



