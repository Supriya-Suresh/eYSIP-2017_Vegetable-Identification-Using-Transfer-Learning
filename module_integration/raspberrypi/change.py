stage = 0
while True:
	data['weight'] = display_screen()
	while True:
		if stage==0:
			imageAccepted,data['imagename'] = takePicture()
			if(imageAccepted):
				stage += 1
			else:
				break
		if stage==1:
			troughIDAccepted,data['troughid'] = acceptTroughID()
			if(troughIDAccepted):
				stage += 1 
			else:
				stage -= 1
				continue
		if stage==2:
			if is_connected(URL_PREDICT):
				success, prediction, cropid = predict(data)
				if success:
					cropid = show_prediction(cropid)
				else:
					while True:
						cropIDAccepted, data['cropid'] =  acceptCropID()
						if(cropIDAccepted):
							 break
						else:
							continue
			else:
				while True:
					cropIDAccepted, data['cropid'] =  acceptCropID()
					if(cropIDAccepted):
						 break
					else:
						continue
			stage += 1
		if stage==3:
			if is_connected(URL):
				send_all_data(data)
			else:
				storeData(data)
			stage = 0